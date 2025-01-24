import rclpy

import copy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

from trajectory import Trajectory, VehicleDescription
from pid_controller import PIDController

from laserscan_filter import LaserscanFilter

from telemetry_monitor_interfaces.msg import Telemetry

import aesthetic_control_interfaces.srv as ae_srv
import aesthetic_control_interfaces.msg as ae_msg

from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
import tf2_geometry_msgs #import required to compute transform!



import math

TOPIC_DRIVE = "/to_drive"
TOPIC_LASERSCAN = "/scan"
TOPIC_ODOMETRY = "/odom"
TOPIC_DEBUG_MARKERARRAY = "/debug/raceline"
TOPIC_DEBUG_MARKER = "/debug/marker"
TOPIC_DEBUG_TELEMETRY = "/debug/telemetry"
TOPIC_LOCALIZATION_COVARIANCE = "/amcl_pose"


class PurePursuit(Node):

    def __init__(self):
        super().__init__('pure_pursuit')
        #create publishers
        self.publisher_ackermann        = self.create_publisher(AckermannDriveStamped, TOPIC_DRIVE, 10)
        self.publisher_markerarray_viz  = self.create_publisher(MarkerArray, TOPIC_DEBUG_MARKERARRAY, 10)
        self.publisher_marker_viz       = self.create_publisher(Marker, TOPIC_DEBUG_MARKER, 10)
        self.publisher_telemetry        = self.create_publisher(Telemetry, TOPIC_DEBUG_TELEMETRY, 10)

        #receive transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        #create listeners
        self.sub_laser = self.create_subscription(LaserScan, TOPIC_LASERSCAN, self.cb_new_laserscan, 1)
        self.sub_laser  # prevent unused variable warning
        self.sub_covariance = self.create_subscription(PoseWithCovarianceStamped, TOPIC_LOCALIZATION_COVARIANCE, self.cb_new_covariance, 1)
        self.sub_covariance  # prevent unused variable warning
        self.__sub_odom     = self.create_subscription(Odometry, TOPIC_ODOMETRY, self.callback_on_odom, 1)
        self.__sub_odom # prevent unused variable warning

        #create services
        """ Deactivate for simulation
        self.__services = {}
        self.__services["underglow"]        = self.create_client(ae_srv.Underglow,      '/carAest/underglow')
        
        #connect to services
        for service in self.__services.values():
            while not service.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'{service} service not available, waiting again...')
        """
        

        #parameters to filter the steering signal before its given to the VESC
        self.pid = PIDController(kp = 1.0, ki = 0.0, kd = 0.0)

        #dimensions of the vehicle
        self.vehicle_width_meters           = 0.28
        self.vehicle_max_steering_angle_deg = 25

        #vehicle state
        self.vehicle_current_velocity = 0.0 #in meters per second
        self.lateral_derivation_history = []


        #raceline
        vd = VehicleDescription()
        self.raceline = Trajectory(x=[0, 1, 2], y=[0, 1, 2], vehicle_description=vd, resolution=0)
        #TODO: file path from configuration or command line

        self.raceline.load_trajectory_from_file("/home/itse/atp_f1tenth_racecar_ws/src/f1tenth_raceline/raceline/my_map_raceline.csv")

        self.lateral_derivation_history = []
        self.last_waypoint_update_velocity = None
        self.enable_veriable_raceline_speed_adaptation = False #let the car change speed of the raceline over time

        self.localization_covariance = [0.0, 0.0, 0.0] #to be updated by the localization algorithm

        self.map_frame_name     = "map"
        self.vehicle_frame_name = "base_link"

        self.max_raceline_speed = max(self.raceline.velocity_profile)

        self.lookahead_m = 0.45           #lookahead to find out steering angle

        self.speed_factor = 0.6          #how much of the speed do we want to apply?
        self.speed_min = 0.6             #minimum speed
        self.speed_max = 3.3             #maximum speed

        self.index_on_raceline = -1      #where on the trajectory are we currently?

        self.max_distance_from_raceline_allowed = 0.35        # if more than so much meters away from the raceline...
        self.max_distance_from_raceline_allowed_penalty = 0.8 # reduce speed with factor

        self.last_steering_angle_rad = 0.0
        self.crash_prevent_steering_angle_rad = None


        self.create_timer(1.0, self.debug_publish_raceline)
        self.create_timer(1.0/30.0, self.dodrive)

    def get_underglow_msg(self, color):
        glow_msg= ae_msg.UnderglowColor()
        glow_msg.set_underglow_color = color
        return glow_msg

    def callback_on_odom(self, msg: Odometry):
        self.vehicle_current_velocity = float( math.sqrt( msg.twist.twist.linear.x **2 + msg.twist.twist.linear.y **2) )

    def debug_publish_raceline(self):
        minvel = min(self.raceline.velocity_profile)
        maxvel = max(self.raceline.velocity_profile)
        color_scale = 1.0/(maxvel-minvel)
        ma = MarkerArray()
        for i in range(len(self.raceline.x)):
            m = Marker()
            m.header.frame_id = self.map_frame_name
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "debug_raceline"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = self.raceline.x[i]
            m.pose.position.y = self.raceline.y[i]
            m.pose.position.z = 0.0
            m.pose.orientation.x = 0.0
            m.pose.orientation.y = 0.0
            m.pose.orientation.z = 0.0
            m.pose.orientation.w = 0.0
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.scale.z = 0.2
            m.color.r = (self.raceline.velocity_profile[i]-minvel)*color_scale
            m.color.g = 1.0-m.color.r
            m.color.b = 0.0
            m.color.a = 1.0

            ma.markers.append(m)
        
        self.publisher_markerarray_viz.publish(ma)

    def dodrive(self):
        try:
            self.drive()
        except Exception as e:
            print(e)

    def transformPoint(self, x:float, y:float, from_frame: str, target_frame:str) -> tuple[float, float]:
        point_in_map = PoseStamped()
        point_in_map.pose.position.x = x
        point_in_map.pose.position.y = y
        point_in_map.header.frame_id = from_frame
        point_in_map.header.stamp = rclpy.time.Time().to_msg() #get the newest transform
        point_in_vehicle = self.tf_buffer.transform(point_in_map, target_frame)
        
        x = point_in_vehicle.pose.position.x
        y = point_in_vehicle.pose.position.y

        return x,y

    def dynamic_lookahead(self, raceline_index):
        factor = self.raceline.velocity_profile[raceline_index] / self.max_raceline_speed

        desired_lookahead = max(self.lookahead_m, 6*factor*self.lookahead_m)

        return desired_lookahead

    def drive(self):
        #find out where we currently are in the map
        try:
            t = self.tf_buffer.lookup_transform(
                                        self.map_frame_name,
                                        self.vehicle_frame_name,
                                        rclpy.time.Time())
            x_vehicle_map = t.transform.translation.x
            y_vehicle_map = t.transform.translation.y
        except Exception as error:
            print(f"No valid transform from {self.map_frame_name} to {self.vehicle_frame_name}. Doin' nothing.", flush=True)
            print(error)
            drive_msg = AckermannDriveStamped()
            drive_msg.header.frame_id = self.vehicle_frame_name
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.drive.steering_angle = float(self.last_steering_angle_rad)
            drive_msg.drive.speed = float(0.0)
            self.publisher_ackermann.publish(drive_msg)
            return


        #check if we have lost tracking of the raceline
        if self.index_on_raceline > 0:
            diff_x = self.raceline.x[self.index_on_raceline]-x_vehicle_map
            diff_y = self.raceline.y[self.index_on_raceline]-y_vehicle_map
            len_diff = math.sqrt(diff_x*diff_x+diff_y*diff_y)

            if len_diff > 2.0:
                self.index_on_raceline = -1


        #find the point of the raceline where we currently are
        best_diff = 100000
        best_idx = self.index_on_raceline
        found_better_point = False
        for i in range(len(self.raceline.x)):
            idx = (self.index_on_raceline + i) % len(self.raceline.x)
            diff_x = self.raceline.x[idx]-x_vehicle_map
            diff_y = self.raceline.y[idx]-y_vehicle_map
            len_diff = math.sqrt(diff_x*diff_x+diff_y*diff_y)
            if len_diff < best_diff:
                #found better point!
                best_diff = len_diff
                best_idx = idx
                found_better_point = True
            else:
                if self.index_on_raceline > -1 and found_better_point:
                    #last time we found a better point - this time not:
                    # we can end the search as from now on, we will be moving away from the best point
                    #break
                    pass
        
        current_distance_from_raceline = best_diff
                
        #lookahead distance should be adaptable to current conditions
        #in curves, it should be smaller, on straights it should be larger
        lookahead = self.dynamic_lookahead(best_idx)



        ###################################
        #find the required steering angle #
        ###################################

        #find next point on trajectory which is self.lookahead away from current position of vehicle
        best_next_idx = best_idx
        best_diff = 100000
        found_better_point = False
        for i in range(len(self.raceline.x)):
            idx = (best_idx + i) % len(self.raceline.x)
            diff_x = self.raceline.x[idx]-x_vehicle_map
            diff_y = self.raceline.y[idx]-y_vehicle_map
            len_diff = abs(lookahead - math.sqrt(diff_x*diff_x+diff_y*diff_y))
            if len_diff < best_diff:
                #found better point!
                best_diff = len_diff
                best_next_idx = idx
                found_better_point = True

                #experimental: make lookahead dynamic based on the velocity profile ahead of the vehicle
                lookahead = min(lookahead, self.dynamic_lookahead(best_next_idx))
            else:
                if found_better_point:
                    #last time we found a better point - this time not:
                    # we can end the search as from now on, we will be moving away from the best point
                    break

        #found the points in map coordinate system. need to transfer back to vehicle coordinate system
        x = self.raceline.x[best_idx]
        y = self.raceline.y[best_idx]

        x, y = self.transformPoint(x, y, self.map_frame_name, self.vehicle_frame_name)

        #DEBUG: publish closest point
        self.publish_point(x, y, 132478, 1.0, 1.0, 1.0, "debug_raceline")

        x = self.raceline.x[best_next_idx]
        y = self.raceline.y[best_next_idx]

        #transform fram map to vehicle coordinate system        
        x, y = self.transformPoint(x, y, self.map_frame_name, self.vehicle_frame_name)


        #publish target point
        self.publish_point(x, y, 132479, 0.0, 0.0, 1.0, "debug_raceline")


        #compute angle to target_point (x,y) (already in vehicle frame)
        b = x
        c = math.sqrt(x*x + y*y)
        alpha = 0.0
        if c != 0.0:
            alpha = math.acos(b/c) #in rad

        #check if we need to steer left or right.
        if y < 0:
            alpha *= -1.0

        
        #use PID filter for steering signal
        alpha = self.pid.update(alpha)

        #find the required speed
        speed = self.raceline.velocity_profile[best_idx] * self.speed_factor

        #slow down when derivated too much from raceline
        if current_distance_from_raceline > self.max_distance_from_raceline_allowed:
            speed *= self.max_distance_from_raceline_allowed_penalty

        #slow down when localization is bad:
        if abs(self.localization_covariance[0]) > 0.4 or abs(self.localization_covariance[1]) > 0.15 or abs(self.localization_covariance[2]) > 0.1:
            speed *= 0.4
            print(f"slowing down: localization bad: {self.localization_covariance}", flush=True)


        #make sure speed is between fixed min and max values
        speed = max(self.speed_min, min(speed, self.speed_max))

        #make sure we do not drive against a wall:
        if self.crash_prevent_steering_angle_rad is not None:
            alpha = self.crash_prevent_steering_angle_rad
            speed = self.speed_min


        #limit the steering angle of the vehicle
        alpha = min(alpha, math.radians(self.vehicle_max_steering_angle_deg))
        alpha = max(alpha, -math.radians(self.vehicle_max_steering_angle_deg))

        #write to VESC
        drive_msg = AckermannDriveStamped()
        drive_msg.header.frame_id = self.vehicle_frame_name
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = float(alpha)
        drive_msg.drive.speed = float(speed)
        self.publisher_ackermann.publish(drive_msg)

        #publish odometry
        #FIXME: lateral derivation only corse guess!
        self.publish_telemetry(next_waypoint_id=best_idx, 
                               pos_x_map=x_vehicle_map, 
                               pos_y_map=y_vehicle_map, 
                               target_velocity=speed, 
                               actual_velocity=self.vehicle_current_velocity, 
                               lateral_derivation=current_distance_from_raceline)
        
        #visualize lateral derivation using the underglow lights
        self.underglow(current_distance_from_raceline)

        #update raceline but only when we are actually driving
        if self.enable_veriable_raceline_speed_adaptation == True:
            if self.vehicle_current_velocity >= self.speed_min:
                self.update_raceline_velocity(current_waypoint=best_idx, 
                                            lateral_derivation=current_distance_from_raceline)
                
        #variable contents for next iteration
        self.index_on_raceline = best_idx
        self.last_steering_angle_rad = alpha


    def update_raceline_velocity(self, current_waypoint, lateral_derivation):
        #if lateral derivation is very low --> increase speed, if too high --> decrease
        history_length = 15
        high_thres = 0.35
        low_thres = 0.2

        if self.last_waypoint_update_velocity == current_waypoint:
            return
        
        #update state for next iteration
        self.last_waypoint_update_velocity = current_waypoint
    
        self.lateral_derivation_history.append( (current_waypoint, lateral_derivation) )
        self.lateral_derivation_history = self.lateral_derivation_history[-history_length:]


        high_count = 0
        low_count = 0
        mid_count = 0

        for wp, ld in self.lateral_derivation_history:
            if ld > high_thres:
                high_count += 1
            if low_thres >= ld <= high_thres:
                mid_count += 1
            if ld < low_thres:
                low_count += 1


        first_wp = self.lateral_derivation_history[0][0]
        last_wp = self.lateral_derivation_history[-1][0]

        num_wp = (len(self.raceline.x) - first_wp) - (len(self.raceline.x) - last_wp)

        if low_count == history_length:
            #increase raceline speed for all history
            for i in range(num_wp):
                wp = first_wp+i % len(self.raceline.x) #to acount for wrap around
                self.raceline.velocity_profile[wp] *= 1.03 # 3% increase

            self.lateral_derivation_history.clear()
        if high_count > 0:
            #decrease raceline speed for all history
            for i in range(num_wp):
                wp = first_wp+i % len(self.raceline.x) #to acount for wrap around
                self.raceline.velocity_profile[wp] = max(self.raceline.velocity_profile[wp]*0.95, self.speed_min) # 5% decrease

            self.lateral_derivation_history.clear()


        

    def underglow(self, current_distance_from_raceline):
        max_dev = 0.4 #all red

        dev_val = min(abs(current_distance_from_raceline), max_dev)
        red = dev_val / max_dev
        green = 1.0-red
        # self.__services["underglow"].call_async(ae_srv.Underglow.Request(glow=self.get_underglow_msg([int(red*255),int(green*255),0])))

    def publish_telemetry(self, next_waypoint_id, pos_x_map, pos_y_map, target_velocity, actual_velocity, lateral_derivation):
        msg = Telemetry()
        msg.next_waypoint_id = next_waypoint_id
        msg.pos_x = pos_x_map
        msg.pos_y = pos_y_map
        msg.target_velocity = target_velocity
        msg.actual_velocity = actual_velocity
        msg.lateral_derivation = lateral_derivation
        self.publisher_telemetry.publish(msg)

    def cb_new_covariance(self, msg: PoseWithCovarianceStamped):
        varx = msg.pose.covariance[0]
        vary = msg.pose.covariance[1]
        varyaw = msg.pose.covariance[35]

        self.localization_covariance = [varx, vary, varyaw]

    def cb_new_laserscan(self, msg: LaserScan):
        #do not drive against walls!
        distance_left  = LaserscanFilter.distance_to_wall(msg, left_wall=True)
        distance_right = LaserscanFilter.distance_to_wall(msg, left_wall=False)

        #todo: make dependent from distance with PID controller!
        if distance_left < self.vehicle_width_meters/2.0:
            self.crash_prevent_steering_angle_rad = math.radians(-10)
            print("Crash Prevention Steering Correction", flush=True)
        else:
            if distance_right < self.vehicle_width_meters/2.0:
                self.crash_prevent_steering_angle_rad = math.radians(10)
                print("Crash Prevention Steering Correction", flush=True)
            else:
                self.crash_prevent_steering_angle_rad = None

    def publish_point(self, x, y, id, r, g, b, namespace):
        m = Marker()
        m.header.frame_id = self.vehicle_frame_name
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = namespace
        m.id = id
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.0
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 0.0
        m.scale.x = 0.5
        m.scale.y = 0.5
        m.scale.z = 0.5
        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = 1.0
        self.publisher_marker_viz.publish(m)

    #destructor
    def __del__(self):
        pass
        #self.stop_vehicle()


def main(args=None):
    rclpy.init(args=args)

    pure_pursuit = PurePursuit()

    rclpy.spin(pure_pursuit)


    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pure_pursuit.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

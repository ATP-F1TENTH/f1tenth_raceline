import random
import math
import pycubicspline.pycubicspline as pyspline
import numpy

class VehicleDescription:
    def __init__(
        self,
        static_friction: float = 0.0, 
        width_m: float = 0.0, 
        mass_kg: float = 0.0,
        acceleration_mss: float = 0.0, 
        deceleration_mss: float = 0.0,
        length_m: float = 0.0,
        min_steering_angle_deg: float = 0.0,
        max_steering_angle_deg: float = 0.0
    ) -> None:
        
        self.haftreibung = static_friction
        self.vehicle_width_m = width_m
        self.vehicle_mass = mass_kg
        self.vehicle_acceleration_mss = acceleration_mss
        self.vehicle_deceleration_mss = deceleration_mss
        self.vehicle_length_m = length_m
        self.vehicle_min_steering_angle = min_steering_angle_deg
        self.vehicle_max_steering_angle = max_steering_angle_deg

class Trajectory:
    def __init__(
        self, x: list,
        y: list,
        vehicle_description: 'VehicleDescription',
        resolution: float,
        curvature=None,
        leave_in_cycle=False,
        is_a_loop=True,
        leave_as_is=False
    ) -> None:
        
        self.x = x
        self.y = y
        self.haftreibung = vehicle_description.haftreibung
        self.vehicle_mass = vehicle_description.vehicle_mass
        self.curvature = curvature
        self.vehicle_width_m = vehicle_description.vehicle_width_m
        self.vehicle_length_m = vehicle_description.vehicle_length_m
        self.vehicle_acceleration_mss = vehicle_description.vehicle_acceleration_mss
        self.vehicle_deceleration_mss = vehicle_description.vehicle_deceleration_mss
        self.vehicle_min_steering_angle = vehicle_description.vehicle_min_steering_angle
        self.vehicle_max_steering_angle = vehicle_description.vehicle_max_steering_angle
        self.resolution = resolution
        self.length = None
        self.laptime = None
        self.is_a_loop = is_a_loop

        self.velocity_profile = None #for each point the velocity of the vehicle in meters/second

        self.do_forwards_pass = False #gives straighter lines as there is more emphasis on acceleration

        if not leave_as_is:
            self.x, self.y, _, self.curvature, _ = pyspline.calc_2d_spline_interpolation(self.x, self.y, num=len(self.y))
            self.remove_overlapping_points(leave_in_cycle=leave_in_cycle)

        #to get a smooth transition from end to start we need the first controlpoint to be the last, as well.
        if is_a_loop:
            self.x.append(self.x[0])
            self.y.append(self.y[0])
            if curvature is not None:
                self.curvature.append(self.curvature[0])

    def get_vehicle_description(self):
        return VehicleDescription(self.haftreibung, self.vehicle_width_m, self.vehicle_mass, self.vehicle_acceleration_mss, self.vehicle_deceleration_mss, self.vehicle_length_m, self.vehicle_min_steering_angle, self.vehicle_max_steering_angle)

    def remove_overlapping_points(self, leave_in_cycle=True):
        """remove points at the end of the spline that overlap with points at the beginning at the spline."""

        if len(self.x) < 7:
            return

        dx = self.x[5]-self.x[6]
        dy = self.y[5]-self.y[6]
        l2 = math.sqrt(dx*dx+dy*dy)
        eps = l2*0.8  #if a point is closer than 80% of the default distance, its considered as a duplicate.
        toRemove = []
        cont_i = False
        for i in range(0, len(self.x)-1):
            for j in range(i+1, len(self.x)):
                if i == j:
                    continue
                dx = self.x[i]-self.x[j]
                dy = self.y[i]-self.y[j]
                l2 = math.sqrt(dx*dx+dy*dy)
                if(l2 < eps):
                    #same point - remove.
                    if not i in toRemove:
                        toRemove.append(i)
                        cont_i = True
                        continue
            if cont_i:
                cont_i = False
                continue

        toRemove.reverse()
        #in order to have a cycle, we need to keep one overlapping point.
        end_idx = len(toRemove)-1       #HERE: changed from -2 to -1
        if not leave_in_cycle:
            end_idx = len(toRemove)-0   #HERE: changed from -1 to 0

        for i in range(0, end_idx):
            self.x.pop(toRemove[i])
            self.y.pop(toRemove[i])
            if self.curvature is not None:
                self.curvature.pop(toRemove[i])


    def adjust_velocity_to_acceleration_backwards_pass(self, i: int):
        """adjust velocity profile to be physically possible"""
        while True:
            velocity = self.velocity_profile[i] #speed at this waypoint (m/s) #TODO: this is not correct: must be average speed, not minimum!!
            xd = self.x[i-1] - self.x[i]
            yd = self.y[i-1] - self.y[i]
            way = math.sqrt(xd*xd+yd*yd)*self.resolution #distance from last to this waypoint (m)
            time = way/velocity # (s)
            if self.velocity_profile[(i-1) % len(self.velocity_profile)]-self.vehicle_deceleration_mss*time > self.velocity_profile[i]:
                self.velocity_profile[(i-1) % len(self.velocity_profile)] = self.velocity_profile[i]+self.vehicle_deceleration_mss*time
                i = (i - 1) % len(self.velocity_profile)
            else:
                break

    def adjust_velocity_to_acceleration_forwards_pass(self, i: int):
        """adjust velocity profile to be physically possible"""
        while True:
            velocity = self.velocity_profile[i] #speed at this waypoint (m/s) #TODO: this is not correct: must be average speed, not minimum!!
            xd = self.x[i] - self.x[(i+1) % len(self.velocity_profile)]
            yd = self.y[i] - self.y[(i+1) % len(self.velocity_profile)]
            way = math.sqrt(xd*xd+yd*yd)*self.resolution #distance from this to next waypoint (m)
            time = way/max(velocity, 0.0001) # (s)
            if self.velocity_profile[(i+1) % len(self.velocity_profile)] > self.velocity_profile[i] + self.vehicle_acceleration_mss*time:
                self.velocity_profile[(i+1) % len(self.velocity_profile)] = self.velocity_profile[i]+self.vehicle_acceleration_mss*time
                i = (i + 1) % len(self.velocity_profile)
            else:
                break

        

    def compute_velocity_profile(self):
        """for each point on the spline, compute the max. possible velocity given a certain traction"""

        # This assumes infinite acceleration/deceleration and gives us the max. possible cornering speeds
        # by computing the antipetal force and letting it always beeing smaller (or equal) to the lateral 
        # force of the tire
        self.velocity_profile = []
        for i in range(0, len(self.x)):
            radius_m = 1.0/max(abs(self.curvature[i]), 0.0001) * self.resolution
            max_velocity = math.sqrt((self.haftreibung * radius_m) / self.vehicle_mass ) #antipetal force
            self.velocity_profile.append(max_velocity)

        #as first and last point have curv. = 0, max velocity does not make sense there
        self.velocity_profile[0]  = self.velocity_profile[1]
        self.velocity_profile[-1] = self.velocity_profile[-2]

        # we now need to resitict speed on the straights such that we have enough braking power to decelerate
        # to this end, find the lowest speed, and go back over the trajectory and decrease speeds if nececcary
        processedList = []

        for numProcessed in range(len(self.x)):
            #find lowest speed and go backwards
            sortedIds = list(numpy.argsort(self.velocity_profile))
            for i in processedList:
                sortedIds.remove(i)

            i = sortedIds[0]

            #go backwards over the trajectory and decrease speeds if nececcary
            self.adjust_velocity_to_acceleration_backwards_pass(i)

            #do the same thing forwards, to compute realistic velocity based on the vehicle acceleration
            if self.do_forwards_pass:
                self.adjust_velocity_to_acceleration_forwards_pass(i)

            processedList.append(i) #point processed
        
    
    def get_length(self) -> int:
        if self.length is not None:
            return self.length
        
        #compute length of spline
        self.length = 0.0
        for i in range(0, len(self.x)-1):
            x2 = self.x[i+1] - self.x[i]
            x2 *= x2
            y2 = self.y[i+1] - self.y[i]
            y2 *= y2
            self.length += math.sqrt( x2 + y2 )

        #add length from last back to first.
        if self.is_a_loop:
            x2 = self.x[-1] - self.x[0]
            x2 *= x2
            y2 = self.y[-1] - self.y[0]
            y2 *= y2
            self.length += math.sqrt( x2 + y2 )

        return self.length

    
    def get_laptime(self, num_samples=200) -> int:
        if self.laptime is not None:
            return self.laptime
        
        if len(self.x) < num_samples/2:
            #make more fine grained trajectory and compute on this
            curvature = None
            if self.is_a_loop:
                lx,ly, _, curvature, _ = pyspline.calc_2d_spline_interpolation(self.x + self.x[1:2], self.y + self.y[1:2], num=num_samples)
            else:
                lx,ly, _, _, _ = pyspline.calc_2d_spline_interpolation(self.x, self.y, num=num_samples)
            rl = Trajectory(lx, ly, self.get_vehicle_description(), self.resolution, is_a_loop=self.is_a_loop, curvature=curvature)
            rl.do_forwards_pass = True

            self.laptime = rl.get_laptime()
            return self.laptime

        #actual computation of laptime based on a trajectory with sufficient points.
        self.compute_velocity_profile()
        #compute length of spline and divide by velocity --> time
        self.laptime = 0.0
        for i in range(0, len(self.x)-1):
            x2 = self.x[i+1] - self.x[i]
            x2 *= x2
            y2 = self.y[i+1] - self.y[i]
            y2 *= y2

            length = math.sqrt( x2 + y2 )
            speed = (self.velocity_profile[i] + self.velocity_profile[i+1])/2.0
            if speed < 0.0001:
                continue
            self.laptime += length/speed

        return self.laptime


    def compute_normal_vector(self, idx: int) -> tuple[float, float]:
        """compute the normalized (orthogonal) normal vector for point with index idx"""
        prev_x = self.x[(idx-1) % len(self.x)]
        prev_y = self.y[(idx-1) % len(self.y)]
        next_x = self.x[(idx+1) % len(self.x)]
        next_y = self.y[(idx+1) % len(self.y)]

        dx = next_x-prev_x
        dy = next_y-prev_y

        orto_dx = -dy
        orto_dy = dx

        length = math.sqrt(orto_dx*orto_dx + orto_dy*orto_dy)

        return orto_dx/length, orto_dy/length
    
    def compute_random_vector(self) -> tuple[float, float]:
        """compute a normalized random vector"""
        dx = 1.0-random.random()*2.0
        dy = 1.0-random.random()*2.0

        length = math.sqrt(dx*dx + dy*dy)

        return dx/length, dy/length

    def random_changes(self, max_change_px: float, num_changes: int, map: list, num_ctrl_points: int, idx=None, apply_smoothing=True, use_normal_vector=False, num_adjacent=4):
        """randomly change the trajectory"""
        change = None
        for i in range(num_changes):
            if idx is None:
                idx = random.randint(0, len(self.x)-2) #exclude last point
            if not (0 <= idx < len(self.x)):
                return 0.0 #externally feed index is out of bounds.
            x = self.x[idx]
            y = self.y[idx]
            #test different changes - keep the first one which is in free space
            while True:
                normalx, normaly = self.compute_random_vector() if use_normal_vector == False else self.compute_normal_vector(idx)
                change = random.random()*max_change_px

                self.x[idx] = x + change*normalx
                self.y[idx] = y + change*normaly
                
                if map[ int(self.y[idx]) ][ int(self.x[idx]) ] == 0.0:
                    #if first point of the trajectory is moved, the last point needs to move, too! (otherwise it wouldn't be a circle anymore)
                    if idx == 0 and self.is_a_loop:
                        self.x[len(self.x)-1] = self.x[idx]
                        self.y[len(self.y)-1] = self.y[idx]
                    break
                else:
                    #revoke changes
                    self.x[idx] = x
                    self.y[idx] = y

            #propergate part of the changes to the surrounding controlpoints, too. to make sure we dont introduce kinks in the raceline
            d = random.randint(1,num_adjacent) #2 #number of adjacent controlpoints which are moved, too
            for j in range(1,d+1):
                if self.is_a_loop:
                    self.x[(idx+j) % len(self.x)] += change*normalx * 1.0/(j+1)
                    self.y[(idx+j) % len(self.x)] += change*normaly * 1.0/(j+1)

                    self.x[(idx-j) % len(self.x)] += change*normalx * 1.0/(j+1)
                    self.y[(idx-j) % len(self.x)] += change*normaly * 1.0/(j+1)
                else:
                    if idx+j < len(self.x):
                        self.x[idx+j] += change*normalx * 1.0/(j+1)
                        self.y[idx+j] += change*normaly * 1.0/(j+1)
                    if idx-j >= 0:
                        self.x[idx-j] += change*normalx * 1.0/(j+1)
                        self.y[idx-j] += change*normaly * 1.0/(j+1)

                #make sure first and last points are the same
                if self.is_a_loop:
                    if (idx+j) % len(self.x) == 0 or (idx-j) % len(self.x) == 0:
                        self.x[len(self.x)-1] = self.x[0]
                        self.y[len(self.y)-1] = self.y[0]
                    if (idx+j) % len(self.x) == len(self.x)-1 or (idx-j) % len(self.x) == len(self.x)-1:
                        self.x[0] = self.x[len(self.x)-1]
                        self.y[0] = self.y[len(self.y)-1]

        #apply spline smoothing
        if apply_smoothing:
            self.x, self.y, _, self.curvature, _ = pyspline.calc_2d_spline_interpolation(self.x, self.y, num=num_ctrl_points)
        self.length = None
        self.laptime = None

        return change

    def random_combination(self, other_trajectory: 'Trajectory', num_ctrl_points: int):
        start_idx = random.randint(0, len(self.x)-1)
        end_idx   = random.randint(start_idx, len(self.x)-1)

        #both trajectories might not have the exact number of points:
        start_idx = min(start_idx, len(other_trajectory.x)-1)
        end_idx = min(end_idx, len(other_trajectory.x)-1)

        for i in range(start_idx, end_idx):
            self.x[i] = other_trajectory.x[i]
            self.y[i] = other_trajectory.y[i]

        #make sure last and first point are the same:
        self.x[0] = self.x[len(self.x)-1]
        self.y[0] = self.y[len(self.y)-1]

        #apply spline smoothing
        self.remove_overlapping_points()
        self.x, self.y, _, self.curvature, _ = pyspline.calc_2d_spline_interpolation(self.x, self.y, num=num_ctrl_points)
        self.remove_overlapping_points()
        self.length = None
        self.laptime = None

    def safe_trajectory_to_file(self, map: 'Map', file: str, num_points: int):
        """ write computed trajectory to file """

        f = open(file, "w")
        f.write("x\ty\tvelocity[m]\n")

        #add second point of spline to end such that the two ends of the spline form a continuous curve
        x,y, _, curvature, _ = pyspline.calc_2d_spline_interpolation(self.x + [self.x[1]], self.y + [self.y[1]], num=num_points)
        
        trajectory = Trajectory(x, y, self.get_vehicle_description(), self.resolution, curvature=curvature)
        trajectory.do_forwards_pass = True
        trajectory.velocity_profile = None
        trajectory.compute_velocity_profile()

        trajectory.remove_overlapping_points(leave_in_cycle=False)

        max_x, max_y = map.get_map_size_pixels()
        resolution = map.get_resolution()
        origin = map.get_origin()

        for i in range(len(trajectory.x)):
            x_px = trajectory.x[i]
            y_px = trajectory.y[i]

            #transform pixel to coordinates - y axis needs to be swapped around!
            x = x_px * resolution + origin[0]
            y = (max_x - y_px) * resolution + origin[1]   #TODO: Shouldn't this be max_y? Somewhere down the line x and y seem to be swapped by mistake!

            f.write(f"{x}\t{y}\t{trajectory.velocity_profile[i]}\t{trajectory.curvature[i]}\n")

        f.close()

    def load_trajectory_from_file(self, file: str):
        """ load precomputed trajectory from file """
        f = open(file, "r")
        f.readline() #ommit first line

        self.x = []
        self.y = []
        self.velocity_profile = []
        self.curvature = []

        for line in f.readlines():
            tokens = line.split("\t")
            self.x.append(float(tokens[0]))
            self.y.append(float(tokens[1]))
            self.velocity_profile.append(float(tokens[2]))
            self.curvature.append(float(tokens[3]))

        f.close()


    def copy(self) -> 'Trajectory':
        t = None
        if self.curvature is not None:
            t = Trajectory(self.x[:], self.y[:], self.get_vehicle_description(), self.resolution, self.curvature[:], is_a_loop=self.is_a_loop)
        else:
            t = Trajectory(self.x[:], self.y[:], self.get_vehicle_description(), self.resolution, is_a_loop=self.is_a_loop)

        t.do_forwards_pass = self.do_forwards_pass
        t.is_a_loop = self.is_a_loop

        return t

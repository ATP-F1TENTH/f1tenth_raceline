
import os
import sys
import yaml
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from matplotlib import pyplot
from pycubicspline.pycubicspline import calc_2d_spline_interpolation as interpolate2d
from trajectory import Trajectory, VehicleDescription
from map import Map

# Define path to config
CONFIG = '/home/itse/atp_f1tenth_racecar_ws/src/f1tenth_raceline/config/optimizer.yaml'

class RacelineOptimizer:
    
    def __init__(self, configfile: str) -> None:

        self.__config = None
        self.__resolution = None

        with open(configfile, 'r') as file:
            try:
                self.__config = yaml.safe_load(file)
            except Exception as e:
                print(e)
                return
            finally:
                file.close()
        
        # reconstruct filepath
        image_path = "/".join(configfile.split("/")[0:-1])
        image_file = image_path + "/" + self.__config["image"]
        self.__resolution = self.__config["resolution"]
        self.__map = Map(image_file=image_file, origin=self.__config["origin"], resolution=self.__resolution)

    @property
    def map(self) -> Map:
        return self.__map

    @property
    def config(self) -> dict:
        return self.__config

    def debug_draw_map(self):
        pyplot.imshow(self.__map.get_pixel_map())
        pyplot.show()

    def debug_draw_trajectory(self, trajectory : Trajectory, filename: str = None, title: str = None):
        pyplot.imshow(self.__map.get_pixel_map())

        lx,ly, _, curvature, _ = interpolate2d(trajectory.x + trajectory.x[1:2], trajectory.y + trajectory.y[1:2], num=300)

        rl = Trajectory(lx, ly, trajectory.get_vehicle_description(), trajectory.resolution, curvature=curvature)
        rl.do_forwards_pass = True
        rl.compute_velocity_profile()
        #print(f"Raceline with 300 points time: {rl.get_laptime()}")

        pyplot.scatter(rl.x,rl.y, c=rl.velocity_profile, linewidth=1, cmap=pyplot.cm.coolwarm)
        pyplot.colorbar()

        if title is not None:
            pyplot.title(title)
        
        try:
            if filename is not None:
                pyplot.savefig(filename, dpi=200)
                pyplot.clf()
            else:
                pyplot.show()
        except:
            print(f"could not save image to file {filename}")

    def get_manual_initial_centerline(self, num_control_points = 200) -> tuple[int, int]:
        """
            Let the user define the initial centerline for optimization by clicking with a mouse.
        """
        print("#####################")
        print("#### click in the image to give manual controlpoints to a spline")
        print("#####################")
        fig = pyplot.figure()
        ax = fig.add_subplot()
        ax.imshow(self.__map.get_pixel_map())
        points = []

        def onclick(event):
            print('add point x=%f, y=%f' % (event.xdata, event.ydata))
            p = [int(event.xdata), int(event.ydata)]
            ax.plot([event.xdata],[event.ydata],"bo")
            fig.canvas.draw()
            points.append(p)

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        pyplot.show()

        #remove all points which are not within free space
        for p in points[:]:
            if self.__map[p[1]][p[0]] > 0.0:
                points.remove(p)

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        #start and end with the same point.
        xs.append(xs[0])
        ys.append(ys[0])

        print(xs)
        print(ys)

        #interpolate:
        x, y, yaw, k, travel = interpolate2d(xs, ys, num=num_control_points)
        
        print(f"Initial Lap Length [m]: {max(travel) * self.__resolution}")

        return x, y

    def optimize_raceline(
        self,
        initial_trajectory: Trajectory,
        turning_radius_m: float, 
        num_epochs=250,
        num_keep=20,
        num_population=200, 
        max_change_in_pixels=3,
        num_changes_per_mutation=1,
        filename="my_map_raceline.csv",
        num_points_file=300, 
        num_ctrl_points=40
    ) -> Trajectory:
        """use a genetic algorithm to find a raceline"""

        def remove_all_but_top(population: list, num_keep: int):
            population.sort(key=lambda x : -1 * x.get_laptime()) #-1 to sort descending!
            return population[len(population)-num_keep : len(population)]

        #initialize population with random racelines deduced from the initial trajectory.
        population = []
        for i in range(num_population):
            t = initial_trajectory.copy()
            t.random_changes(max_change_in_pixels, num_changes_per_mutation, self.__map, num_ctrl_points=num_ctrl_points)
            population.append(t)
        population.append(initial_trajectory.copy()) #keep in the original one w/o modifications

        #remove all but the top racelines
        population = remove_all_but_top(population, num_keep)
        
        #go through epochs
        for e in range(0, num_epochs):

            #create new offspring
            new_childs = []
            offspring_count = int(math.ceil(num_population/num_keep))
            for rl in population:
                for i in range(offspring_count):
                    new_childs.append(rl.copy())

            #mutate offspring
            for rl in new_childs:
                rl.random_changes(max_change_in_pixels, num_changes_per_mutation, self.__map, num_ctrl_points=num_ctrl_points)
            
            new_childs += population[:] #copy over old trajectories to new childs for random_combination
            
            #combine a random pair of trajectories
            combined_childs = []
            """random.shuffle(new_childs)
            for i in range(0, int(len(new_childs)/2)):
                i = random.randint(0, len(new_childs)-1)
                j = random.randint(0, len(new_childs)-1)
                combined = new_childs[i].copy()
                combined.random_combination(new_childs[j], num_ctrl_points=num_ctrl_points)
                combined_childs.append( combined )
            """

            #add the newly mutated children to population:
            for l in new_childs:
                population.append(l)
            for l in combined_childs:
                population.append(l)

            #make sure population are NOT driving through non-free space!
            #and that turning radius is feasable for the vehicle
            max_curvature = 1.0 / (turning_radius_m/population[0].resolution)
            vehicle_width_in_map_pixels = math.ceil(population[0].vehicle_width_m / self.__resolution)
            for l in population[:]:
                lx,ly, _, curvature, _ = interpolate2d(l.x, l.y, num=500)

                #check curvature:
                curvature_ok = True
                for c in curvature:
                    if abs(c) > max_curvature:
                        curvature_ok = False
                        break
                
                if not curvature_ok:
                    population.remove(l)
                    continue

                #for each point of the trajectory:
                for i in range(len(lx)):
                    #check if a square around each point of the trajectory is all in free space
                    #TODO: This should actually be a circle!
                    removeTrajectory = False
                    if self.__map[int(ly[i])][int(lx[i])] > 0.0:
                        #print(f"Point {i} not in free space: {int(lx[i])},{int(ly[i])}")
                        removeTrajectory = True
                        break
                    for dx in range(-math.floor(vehicle_width_in_map_pixels/2.0), math.ceil(vehicle_width_in_map_pixels/2.0), 1):
                        for dy in range(-math.floor(vehicle_width_in_map_pixels/2.0), math.ceil(vehicle_width_in_map_pixels/2.0), 1):
                            if self.__map[int(ly[i]+dy)][int(lx[i]+dx)] > 0.0:
                                #print(f"Point {i} not in free space: {int(ly[i]+dy)},{int(lx[i]+dx)}")
                                removeTrajectory = True
                                break
                        if removeTrajectory:
                            break
                    if removeTrajectory:
                            break

                if removeTrajectory:
                    population.remove(l)

            if len(population) == 0:
                raise Exception("Not possible to find a raceline. Check parameters and initial raceline.")
            
            print(f"Valid Population size: {len(population)}")

            #remove all but the top racelines
            population = remove_all_but_top(population, num_keep)

            #print length of best raceline.
            laptime_s = population[-1].get_laptime() * self.map.get_resolution()
            print(f"raceline length/laptime in epoch {e}: {population[-1].get_length() * self.map.get_resolution()} / {laptime_s}")
            self.debug_draw_trajectory(population[-1], f"racelines/racelineEpoch{e}.png", title=f"Epoch {e} - Laptime {laptime_s}s")
            population[-1].safe_trajectory_to_file(self.map, filename, num_points=num_points_file)
            
        
        population[-1].safe_trajectory_to_file(self.map, filename, num_points=num_points_file)
        return population[-1]
    

def main(config_file: str = None):

    # load yaml config and get subclasses
    with open(config_file, 'r') as file:
        try:
            config_parameter = yaml.safe_load(file)['optimization_parameter']
            vehicle_parameter = config_parameter['vehicle_settings']
            raceline_settings = config_parameter['raceline_settings']
            evolution_settings = config_parameter['evolution_settings']
        except Exception as e:
            print(e)

    # create optimizer instance
    map_path = raceline_settings['map_path']
    opt = RacelineOptimizer(map_path)
    map_resolution = opt.config['resolution']
    
    # for development: fixed start trajectory - in production this should come from waypoints sampled from follow the gap algorithm.
    x, y = opt.get_manual_initial_centerline()

    # Set initial point and threshold for racetrack extraction in map settings file
    with open(map_path, 'r') as file:
        settings = yaml.safe_load(file)
    settings['extraction_settings'] = dict()
    settings['extraction_settings']['start_coordinates'] = [(int(x[0]), int(y[0]))]
    settings['extraction_settings']['threshold'] = 210
    with open(map_path, 'w') as file:
        file.write(yaml.safe_dump(settings))

    # add splines first point also as last one -> create loop
    x.append(x[0])
    y.append(y[0])
    x, y, _, _, path_len = interpolate2d(x, y, num=raceline_settings['num_interpolation_points'])
    
    # compute number of control points
    track_in_meters = path_len[-1] * map_resolution
    num_ctrl_points = math.ceil(track_in_meters * raceline_settings['desired_points_per_meter'])

    # resample spline with desired number of control points
    x, y, _, _, path_len = interpolate2d(x, y, num=num_ctrl_points)

    # create VehicleDescription and Trajectory
    vd = VehicleDescription(**vehicle_parameter)
    original = Trajectory(x, y, vd, map_resolution)

    # use genetic algorithm to optimize x, y
    raceline = opt.optimize_raceline(
        **evolution_settings,
        initial_trajectory=original,
        max_change_in_pixels=raceline_settings['max_change_per_point_meters'] / map_resolution,
        filename=raceline_settings['csv_file_name'],
        num_ctrl_points=num_ctrl_points
    )

    raceline.laptime = None
    raceline.do_forwards_pass = True
    print(f"Raceline time: {raceline.get_laptime()}")
    opt.debug_draw_trajectory(raceline, title="")


if __name__ == "__main__":
    args = sys.argv
    if (len(args) == 2 and args[1] =='--config'):
        main(config_file=args[2])
    else:
        main(config_file=CONFIG)
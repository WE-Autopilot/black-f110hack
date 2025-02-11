import numpy as np
from PIL import Image
import time
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
from argparse import Namespace

# Define the PurePursuitPlanner class in your script
class PurePursuitPlanner:
    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.0
        self.drawn_waypoints = []

    def load_waypoints(self, conf):
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def render_waypoints(self, e):
        points = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        scaled_points = 50.0 * points
        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.0]),
                               ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.0]

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i + t, wrap=True)
            if i2 is None:
                return None
            current_waypoint = np.empty((3,))
            current_waypoint[0:2] = wpts[i2, :]
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)
        if lookahead_point is None:
            return 4.0, 0.0
        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed
        return speed, steering_angle

def get_car_view(map_image, obs, crop_width=200, crop_height=200):
    """
    Returns a cropped and rotated image centered on the car's position from the F1tenth gym environment.

    :param map_image: PIL Image of the racetrack map.
    :param obs: Observation dictionary from F1tenth gym environment.
    :param crop_width: Width of the cropped image.
    :param crop_height: Height of the cropped image.
    :return: Cropped and rotated PIL Image.
    """
    # Extract car position and orientation from the first agent in obs
    x = obs['poses_x'][0]
    y = obs['poses_y'][0]
    theta = obs['poses_theta'][0]

    # Convert theta from radians to degrees for PIL's rotation
    theta_deg = -np.degrees(theta)  # Negative because PIL rotates counterclockwise

    # Define the region of interest (ROI) around the car's position
    left = x - crop_width // 2
    top = y - crop_height // 2
    right = x + crop_width // 2
    bottom = y + crop_height // 2

    # Crop the image
    cropped_image = map_image.crop((int(left), int(top), int(right), int(bottom)))

    # Rotate the image around its center
    rotated_image = cropped_image.rotate(theta_deg, center=(crop_width // 2, crop_height // 2))

    return rotated_image

def main():
    """
    Main function to run the F1tenth gym simulation and save car view images.
    """
    # Define the work parameters
    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 1.375}

    # Load the configuration file
    with open('berlin.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # Initialize the planner
    planner = PurePursuitPlanner(conf, (0.17145 + 0.15875))

    # Load the map image
    map_image = Image.open(conf.map_path)

    # Initialize the gym environment
    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)

    # Reset the environment
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()

    laptime = 0.0
    start = time.time()

    # Main simulation loop
    while not done:
        # Plan the next action
        speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
        
        # Get and save the car view image
        car_view = get_car_view(map_image, obs)
        car_view.save(f'car_view_lap_{laptime:.2f}.png')

        # Step the environment
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward
        env.render(mode='human')

    # Print elapsed time
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)

if __name__ == '__main__':
    main()
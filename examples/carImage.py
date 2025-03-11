import numpy as np
from PIL import Image
import time
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
from argparse import Namespace

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
    with open('config_example_map.yaml') as file:
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
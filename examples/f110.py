import time
import yaml
import gym
import numpy as np
from argparse import Namespace

from f110_gym.envs.base_classes import Integrator

from pilot import PurePursuitPlanner

def main(render_on=True):
    """
    Main entry point for running the F110 environment with a chosen planner.
    Set render_on=False to disable rendering and save resources.
    """
    # Example parameter set
    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}

    '''
    work = {'mass': 3.463388126201571,
            'lf': 0.05597534362552312,
            'tlad': 0.72461887897713965,
            'vgain': 1.375}
    '''

    # Load config from YAML
    with open('assets/config.yaml') as file:
        conf_dict = yaml.safe_load(file)
    conf = Namespace(**conf_dict)

    # Choose your planner here
    planner = PurePursuitPlanner(conf, wheelbase=(0.17145 + 0.15875))

    # Optional rendering callback
    def render_callback(env_renderer):
        # Only do extra rendering if we actually want to see something
        if render_on:
            # Update camera to follow the car
            e = env_renderer
            x = e.cars[0].vertices[::2]
            y = e.cars[0].vertices[1::2]
            top, bottom, left, right = max(y), min(y), min(x), max(x)
            e.score_label.x = left
            e.score_label.y = top - 700

            # Control how much is visible on the screen
            e.left = left - 400
            e.right = right + 400
            e.top = top + 400
            e.bottom = bottom - 400

            # Render waypoints for the chosen planner
            planner.render_waypoints(env_renderer)

    # Create environment
    env = gym.make('f110_gym:f110-v0',
                   map=conf.map_path,
                   map_ext=conf.map_ext,
                   num_agents=1,
                   timestep=0.01,
                   integrator=Integrator.RK4)

    # Register the render callback if we are rendering
    if render_on:
        env.add_render_callback(render_callback)

    # Reset environment
    obs, step_reward, done, info = env.reset(
        np.array([[conf.sx, conf.sy, conf.stheta]])
    )

    # Render the environment if desired
    if render_on:
        env.render(mode='human')

    laptime = 0.0
    start = time.time()

    # Main loop
    while not done:
        speed, steer = planner.plan(
            obs['poses_x'][0],
            obs['poses_y'][0],
            obs['poses_theta'][0],
            work['tlad'],
            work['vgain']
        )
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward

        # Only render if the toggle is on
        if render_on:
            env.render(mode='human')

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)


if __name__ == '__main__':
    # Example usage:
    main(render_on=True)  # set to False to disable UI rendering

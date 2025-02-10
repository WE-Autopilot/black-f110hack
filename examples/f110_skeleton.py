import time
import yaml
import gym
import numpy as np
from argparse import Namespace
from f110_gym.envs.base_classes import Integrator
from pyglet.gl import GL_LINES  # For drawing arrows

def get_waypoints():
    """
    This function will later receive input from the SAL model.
    For now, it generates spaced-out waypoints.
    """
    x_vals = np.linspace(0, 50, 64)  # More spaced out x-coordinates
    y_vals = np.sin(x_vals / 5.0) * 10  # Example curving path

    waypoints = np.vstack((x_vals, y_vals)).T  # Shape (64,2)

    # Print waypoints for debugging
    print("Generated Waypoints (64x2):")
    print(waypoints)

    return waypoints

def main():
    # 1) Load configuration for map, start pose, etc.
    with open('config_example_map.yaml', 'r') as f:
        conf_dict = yaml.load(f, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # 2) Get waypoints from function
    waypoints = get_waypoints()  # This will later be replaced by the SAL model's output

    # 3) Create the F1TENTH environment
    env = gym.make(
        'f110_gym:f110-v0',
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4
    )

    # 4) Render waypoints as arrows
    drawn_arrows = []

    def render_callback(env_renderer):
        e = env_renderer
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left   = left   - 400
        e.right  = right  + 400
        e.top    = top    + 400
        e.bottom = bottom - 400

        # Scale waypoints for visualization
        scale_factor = 10.0
        scaled_waypoints = scale_factor * (waypoints - np.mean(waypoints, axis=0))

        # Start arrow at car position
        car_x, car_y = np.mean(x), np.mean(y)
        first_arrow = np.array([[car_x, car_y], scaled_waypoints[0]])

        # Create arrow paths (tip-to-tail)
        arrows = [first_arrow] + [
            np.array([scaled_waypoints[i], scaled_waypoints[i + 1]])
            for i in range(len(scaled_waypoints) - 1)
        ]

        for i in range(len(arrows)):
            start, end = arrows[i]
            if len(drawn_arrows) < len(arrows):
                b = e.batch.add(
                    2, GL_LINES, None,
                    ('v3f/stream', [start[0], start[1], 0.0, end[0], end[1], 0.0]),
                    ('c3B/stream', [255, 0, 0, 255, 0, 0])  # Red color
                )
                drawn_arrows.append(b)
            else:
                drawn_arrows[i].vertices = [start[0], start[1], 0.0, end[0], end[1], 0.0]

    env.add_render_callback(render_callback)

    # 5) Reset environment
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()

    # 6) Main simulation loop
    total_time = 0.0
    start_time = time.time()

    while not done:
        steer = 0.0
        speed = 1.0
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        total_time += step_reward
        env.render(mode='human')

    print("Simulated Lap Time:", total_time)
    print("Wall-Clock Time:", time.time() - start_time)

if __name__ == '__main__':
    main()

import time
import yaml
import gym
import numpy as np
from argparse import Namespace
from f110_gym.envs.base_classes import Integrator
from pyglet.gl import GL_POINTS  # optional if you want rendering

def main():
    # 1) Load configuration for map, start pose, etc.
    with open('config_example_map.yaml', 'r') as f:
        conf_dict = yaml.load(f, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # 2) Create the F1TENTH environment
    env = gym.make(
        'f110_gym:f110-v0',
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4
    )

    # 3) Optional render callback for camera or custom drawing
    def render_callback(env_renderer):
        e = env_renderer
        # Example camera logic that follows the first car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left   = left   - 400
        e.right  = right  + 400
        e.top    = top    + 400
        e.bottom = bottom - 400
        # No waypoints to draw here, but you could add custom shapes if needed

    env.add_render_callback(render_callback)

    # 4) Reset environment with initial pose (from config)
    obs, step_reward, done, info = env.reset(
        np.array([[conf.sx, conf.sy, conf.stheta]])
    )
    env.render()

    # 5) Main simulation loop
    total_time = 0.0
    start_time = time.time()

    while not done:
        # ---- Dummy action: fixed or random. Replace with ML policy later ----
        
        """
        If we were to replace this later we can do something like:
        steer, speed = rl_model.predict(obs)
        """
        steer = 0.0
        speed = 1.0

        # Step the environment
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        total_time += step_reward

        # Render
        env.render(mode='human')

    print("Simulated Lap Time:", total_time)
    print("Wall-Clock Time:", time.time() - start_time)

if __name__ == '__main__':
    main()

# MIT License
# Full license text here...

import cv2
import os
import math
import numpy as np
import shapely.geometry as shp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Seed for the numpy RNG.')
parser.add_argument('--num_maps', type=int, default=1, help='Number of maps to generate.')
parser.add_argument('--num_obstacles', type=int, default=10, help='Number of obstacles to generate.')
args = parser.parse_args()

np.random.seed(args.seed)

if not os.path.exists('maps'):
    print('Creating maps/ directory.')
    os.makedirs('maps')
if not os.path.exists('centerline'):
    print('Creating centerline/ directory.')
    os.makedirs('centerline')

NUM_MAPS = args.num_maps
WIDTH = 10.0
OBSTACLE_RADIUS_MIN = 1.0
OBSTACLE_RADIUS_MAX = 2.0


def create_track():
    """
    Generates a random closed-loop track with checkpoints.
    """
    CHECKPOINTS = 16
    SCALE = 6.0
    TRACK_RAD = 900 / SCALE
    TRACK_DETAIL_STEP = 21 / SCALE
    TRACK_TURN_RATE = 0.31

    # Create checkpoints
    checkpoints = []
    for c in range(CHECKPOINTS):
        alpha = 2 * math.pi * c / CHECKPOINTS
        rad = np.random.uniform(TRACK_RAD / 3, TRACK_RAD)
        checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))

    # Smooth the track using linear interpolation
    track = []
    for i in range(len(checkpoints)):
        p1 = checkpoints[i]
        p2 = checkpoints[(i + 1) % len(checkpoints)]
        for t in np.linspace(0, 1, int(TRACK_DETAIL_STEP)):
            alpha = (1 - t) * p1[0] + t * p2[0]
            x = (1 - t) * p1[1] + t * p2[1]
            y = (1 - t) * p1[2] + t * p2[2]
            track.append((alpha, x, y))

    # Convert to numpy array
    track = np.array([[t[1], t[2]] for t in track])

    # Validate the track geometry
    track_poly = shp.Polygon(track)
    if not track_poly.is_valid:
        print("Invalid track geometry. Retrying...")
        return None

    # Create inner and outer boundaries
    track_int = np.array(track_poly.buffer(-WIDTH).exterior.coords)
    track_ext = np.array(track_poly.buffer(WIDTH).exterior.coords)

    return track, track_int, track_ext


def generate_obstacles(track_int, track_ext, num_obstacles):
    """
    Generates obstacles randomly within the track boundaries.
    """
    obstacles = []
    track_poly = shp.Polygon(track_ext).difference(shp.Polygon(track_int))

    for _ in range(num_obstacles):
        while True:
            x = np.random.uniform(track_poly.bounds[0], track_poly.bounds[2])
            y = np.random.uniform(track_poly.bounds[1], track_poly.bounds[3])
            if track_poly.contains(shp.Point(x, y)):
                radius = np.random.uniform(OBSTACLE_RADIUS_MIN, OBSTACLE_RADIUS_MAX)
                obstacles.append((x, y, radius))
                break
    return obstacles


def convert_track_with_waypoints(track, track_int, track_ext, iter, obstacles):
    """
    Converts track to image, saves centerline and boundaries as CSV and YAML files, and adds obstacles to the map.
    """
    if track_int.size == 0 or track_ext.size == 0:
        print(f"Empty track boundaries for map {iter}. Skipping...")
        return

    # Plot the track and obstacles
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 20)
    ax.plot(track[:, 0], track[:, 1], color='blue', linewidth=2, label='Centerline')
    ax.plot(track_int[:, 0], track_int[:, 1], color='black', linewidth=3, label='Inner Boundary')
    ax.plot(track_ext[:, 0], track_ext[:, 1], color='black', linewidth=3, label='Outer Boundary')

    # Add obstacles as red circles
    for x, y, r in obstacles:
        circle = Circle((x, y), r, color='red', alpha=0.6)
        ax.add_patch(circle)

    plt.tight_layout()
    ax.set_aspect('equal')
    plt.legend()
    plt.savefig(f'maps/map{iter}_obstacles.png', dpi=80)
    plt.close()

    # Save the waypoints (centerline) as a CSV file with speeds
    speed = 1.0  # Default speed value
    with open(f'centerline/map{iter}.csv', 'w') as csv_file:
        csv_file.write("x,y,speed\n")  # CSV header
        for x, y in track:
            csv_file.write(f"{x},{y},{speed}\n")

    # Create a YAML configuration file
    yaml_data = {
        'map_path': f'maps/map{iter}_obstacles.png',
        'map_ext': '.png',
        'sx': track[0, 0],  # Start x
        'sy': track[0, 1],  # Start y
        'stheta': 0.0,      # Start theta
        'wpt_path': f'centerline/map{iter}.csv',
        'wpt_delim': ',',
        'wpt_rowskip': 1,
        'wpt_xind': 0,
        'wpt_yind': 1,
        'wpt_vind': 2
    }
    with open(f'maps/map{iter}.yaml', 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file)

    print(f"Saved map {iter}: obstacles, YAML, and CSV files.")


if __name__ == '__main__':
    for i in range(NUM_MAPS):
        try:
            result = create_track()
            if result is None:
                continue
            track, track_int, track_ext = result

            # Generate obstacles
            obstacles = generate_obstacles(track_int, track_ext, args.num_obstacles)

            # Convert and save track with waypoints
            convert_track_with_waypoints(track, track_int, track_ext, i, obstacles)

        except Exception as e:
            print(f"Unexpected error during track generation: {e}")
            continue

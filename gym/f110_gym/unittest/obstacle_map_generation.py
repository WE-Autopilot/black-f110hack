
# PLEASE NOTE THE FOLLOWING:
# You will need the necessary dependencies to run this program
# This includes Shapely, matplotlib, numpy, etc
# To run the program, you need the following command
# python .\random_trackgen.py --seed 123 --num_maps arg --num_obstacles arg
# Replace arg with positive integers, all maps and centerline data are found in their folders under unittest

import cv2
import os
import math
import numpy as np
import shapely.geometry as shp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

 # For testing data, we can adjust the sizes of obstacles to be whatever we need   

NUM_MAPS = args.num_maps
WIDTH = 1.0  # Track width (1.0 meter, as per F1TENTH specifications)
CAR_LENGTH = 0.3  # Car length (0.3 meters, as per F1TENTH specifications)
CAR_WIDTH = 0.2   # Car width (0.2 meters, as per F1TENTH specifications)
OBSTACLE_WIDTH_MIN = 0.2
OBSTACLE_WIDTH_MAX = 0.5
OBSTACLE_HEIGHT_MIN = 0.2
OBSTACLE_HEIGHT_MAX = 0.8
CLEARANCE = CAR_WIDTH + 0.1  # Minimum clearance required for the car to navigate around obstacles (10 cm buffer)


def create_track():
    """
    Generates a random closed-loop track with checkpoints.
    """
    CHECKPOINTS = 16
    SCALE = 6.0
    TRACK_RAD = 90 / SCALE
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


def generate_obstacles(track, track_int, track_ext, num_obstacles):
    """
    Generates obstacles ensuring the track remains navigable.
    - Center obstacles are placed randomly on waypoints.
    - Wall obstacles are placed along the inner or outer boundaries.
    - Ensures obstacles do not block the track.
    """
    obstacles = []
    track_poly = shp.Polygon(track_ext).difference(shp.Polygon(track_int))

    total_obstacle_area = 0  # Track total area covered by obstacles
    MAX_OBSTACLE_AREA_RATIO = 0.3  # Obstacles can cover up to 30% of track area
    MIN_GAP = CAR_WIDTH + 0.2 # This ensures 1 car length is able to get through
    # Create LineStrings for track boundaries
    inner_line = shp.LineString(track_int)
    outer_line = shp.LineString(track_ext)

    for _ in range(num_obstacles):
        placement_attempts = 20  # Maximum retries for placing an obstacle, if it cant find a spot we disregard it
        placed = False

        while placement_attempts > 0 and not placed:
            # Randomly decide whether to place in center or near a wall
            place_near_wall = np.random.choice([True, False])

            if place_near_wall:
                # Place obstacle near a track boundary (inner or outer wall)
                boundary = np.random.choice([inner_line, outer_line])
                distance = np.random.random() * boundary.length
                point = boundary.interpolate(distance)
                x, y = point.x, point.y

                # Generate width and height
                width = np.random.uniform(OBSTACLE_WIDTH_MIN, min(OBSTACLE_WIDTH_MAX, WIDTH - CLEARANCE))
                height = np.random.uniform(OBSTACLE_HEIGHT_MIN, min(OBSTACLE_HEIGHT_MAX, WIDTH - CLEARANCE))

                # Move obstacle slightly away from the boundary
                normal = np.array([x - track[0][0], y - track[0][1]])
                normal = normal / np.linalg.norm(normal)
                
                x += width * 0.3 * normal[0]
                y += height * 0.3 * normal[1]


            else:
                # Place obstacle at a random waypoint (center track placement)
                waypoint_idx = np.random.randint(0, len(track))
                x, y = track[waypoint_idx]

                # Generate width and height
                width = np.random.uniform(OBSTACLE_WIDTH_MIN, min(OBSTACLE_WIDTH_MAX, WIDTH - CLEARANCE))
                height = np.random.uniform(OBSTACLE_HEIGHT_MIN, min(OBSTACLE_HEIGHT_MAX, WIDTH - CLEARANCE))

            # Ensure the obstacle doesn’t block too much of the track
            if total_obstacle_area + (width * height) > MAX_OBSTACLE_AREA_RATIO * track_poly.area:
                print("Skipping obstacle to prevent excessive blockage.")
                break  # Stop placing obstacles if total area limit is reached

            # Create the obstacle polygon
            obstacle_poly = shp.Polygon([
                (x - width / 2, y - height / 2),
                (x - width / 2, y + height / 2),
                (x + width / 2, y + height / 2),
                (x + width / 2, y - height / 2)
            ])

            # Ensure there’s still a continuous path
            remaining_space = track_poly.difference(obstacle_poly.buffer(CLEARANCE))

            if remaining_space.is_empty or remaining_space.geom_type == 'MultiPolygon':
                # Tries to reduce size so it guarentees a path
                width *= 0.8
                height *= 0.8
                placement_attempts -= 1
                continue  

            # Ensure a minimum gap is left between obstacles
            valid_placement = True
            for ox, oy, ow, oh in obstacles:
                dist = math.sqrt((x - ox) ** 2 + (y - oy) ** 2)
                if dist < MIN_GAP:
                    valid_placement = False
                    break
            # Valid placement check
            if valid_placement:
                obstacles.append((x, y, width, height))
                total_obstacle_area += width * height
                print(f"Placed {'wall' if place_near_wall else 'center'} obstacle at ({x:.2f}, {y:.2f}) with size {width:.2f}x{height:.2f}")
                placed = True
            else:
                # Try shifting position slightly
                x += np.random.uniform(-0.2, 0.2)
                y += np.random.uniform(-0.2, 0.2)
                placement_attempts -= 1

        if not placed:
            print("Failed to place an obstacle after multiple attempts, skipping.")

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
    ax.plot(track_int[:, 0], track_int[:, 1], color='black', linewidth=1, label='Inner Boundary')  # Thinner lines
    ax.plot(track_ext[:, 0], track_ext[:, 1], color='black', linewidth=1, label='Outer Boundary')  # Thinner lines

    # Add obstacles as pink squares
    for x, y, width, height in obstacles:
        rect = Rectangle((x - width / 2, y - height / 2), width, height, 
                        facecolor='#000000',  # Changed color
                        alpha=0.8,  # More opaque
                        linewidth=1.5)  # Thicker border
        ax.add_patch(rect)

    plt.tight_layout()
    ax.set_aspect('equal')
    plt.axis('off')
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

            # Generate square obstacles
            obstacles = generate_obstacles(track, track_int, track_ext, args.num_obstacles)

            # Convert and save track with waypoints
            convert_track_with_waypoints(track, track_int, track_ext, i, obstacles)

        except Exception as e:
            print(f"Unexpected error during track generation: {e}")
            continue
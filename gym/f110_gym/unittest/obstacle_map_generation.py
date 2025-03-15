#ROSA'S CODE
# PLEASE NOTE THE FOLLOWING:
# You will need the necessary dependencies to run this program
# This includes Shapely, matplotlib, numpy, etc
# To run the program, you need the following command
# python .\obstacle_generation.py --seed 123 --num_maps arg --num_obstacles arg
# Replace arg with positive integers, all maps and centerline data are found in their files under unittest

import cv2
import os
import math
import numpy as np
import shapely.geometry as shp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Seed for the numpy RNG.')
parser.add_argument('--num_maps', type=int, default=1, help='Number of maps to generate.')
parser.add_argument('--num_obstacles', type=int, default=10, help='Number of obstacles to generate.')
parser.add_argument('--base_path', type=str, default=None, help='Base path for absolute file paths in YAML.')
parser.add_argument('--origin_x', type=float, default=11.964691855978616, help='X-coordinate for origin in YAML.')
parser.add_argument('--origin_y', type=float, default=0.0, help='Y-coordinate for origin in YAML.')
parser.add_argument('--origin_theta', type=float, default=1.0, help='Theta value for origin in YAML.')
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
WIDTH = 10.0  # Track width (10.0 meters)
CAR_LENGTH = 3.0  # Car length (3.0 meters)
CAR_WIDTH = 2.0   # Car width (2.0 meters)
OBSTACLE_WIDTH_MIN = 2.5
OBSTACLE_WIDTH_MAX = 8.0
OBSTACLE_HEIGHT_MIN = 6.0
OBSTACLE_HEIGHT_MAX = 8.0
CLEARANCE = CAR_WIDTH + 1.0  # Minimum clearance required for the car to navigate around obstacles

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

            # Ensure the obstacle doesn't block too much of the track
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

            # Ensure there's still a continuous path
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
    ax.set_aspect('equal')

    # Fill the background with black (this is so we see black as illegal and white (the track) being legal)
    ax.set_facecolor("black")  

    # Fill the area outside the track with black, leaving the track white
    outer_boundary = np.vstack([track_ext, track_ext[0]])  # Ensure closed shape
    inner_boundary = np.vstack([track_int, track_int[0]])  # Ensure closed shape

    # This just fills the inside of the track with white, takes into consideration inner and outer boundaries and fills them accordingly
    outer_polygon = Polygon(outer_boundary, facecolor='white', edgecolor='white', linewidth=1)
    inner_polygon = Polygon(inner_boundary, facecolor='black', edgecolor='black', linewidth=1)
    
    ax.add_patch(outer_polygon)  # Add white track
    ax.add_patch(inner_polygon)  # Carves out the inside of the track

    # Plot the track boundaries in white
    ax.plot(track_int[:, 0], track_int[:, 1], color='black', linewidth=1)  
    ax.plot(track_ext[:, 0], track_ext[:, 1], color='black', linewidth=1)  

    # Hide axes (we dont want those as we just want the track)
    plt.axis('off')
    # Change the axes color to white
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['top'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.gca().spines['right'].set_color('white')

    # Change the tick color to white (if needed)
    plt.tick_params(colors='white')

    # Add obstacles as black squares
    for x, y, width, height in obstacles:
        rect = Rectangle((x - width / 2, y - height / 2), width, height, 
                        facecolor='#000000',  # Changed color
                        alpha=1.0,  # More opaque
                        linewidth=1.5)  # Thicker border
        ax.add_patch(rect)

    plt.tight_layout()
    
    # Save the map as grayscale image
    plt.savefig(f'maps/map{iter}.png', dpi=80, facecolor="black")
    
    # Save a debug version with waypoints
    plt.plot(track[:, 0], track[:, 1], 'ro', markersize=2)  # Plot waypoints
    plt.savefig(f'maps/map{iter}_debug.png', dpi=80, facecolor="black")
    plt.close()

    # Save the waypoints (centerline) as a CSV file with speeds and theta
    speed = 1.0  # Default speed value
    
    # Calculate theta (heading angle) for each waypoint
    thetas = []
    for i in range(len(track)):
        # Get current and next point (wrapping around to start for the last point)
        current = track[i]
        next_pt = track[(i + 1) % len(track)]
        
        # Calculate vector and angle
        dx = next_pt[0] - current[0]
        dy = next_pt[1] - current[1]
        theta = np.arctan2(dy, dx)  # Returns angle in radians
        thetas.append(theta)
    
    # Write to CSV with theta included
    with open(f'centerline/map{iter}.csv', 'w') as csv_file:
        csv_file.write("x;y;speed;theta\n")  # CSV header
        for i, (x, y) in enumerate(track):
            csv_file.write(f"{x};{y};{speed};{thetas[i]}\n")

    # Determine paths (relative or absolute)
    if args.base_path:
        map_path = os.path.join(args.base_path, f'../assets/maps/map{iter}')
        wpt_path = os.path.join(args.base_path, f'../assets/maps/map{iter}.csv')
    else:
        map_path = f'../assets/maps/map{iter}'
        wpt_path = f'../assets/maps/map{iter}.csv'

    start_x, start_y = track[0]
    start_theta = thetas[0]

    # Create a YAML configuration file with the desired format
    yaml_data = {
        'resolution': 0.1,  # meters per pixel
        'origin': [args.origin_x, args.origin_y, args.origin_theta],  # x, y, theta (in meters and radians)
        'map_path': map_path,
        'map_ext': '.png',
        'sx': float(start_x),  # starting points for sx
        'sy': float(start_y),  # starting points  for sy
        'stheta': float(start_theta),
        'wpt_path': wpt_path,
        'wpt_delim': ';',
        'wpt_rowskip': 1,
        'wpt_xind': 1,
        'wpt_yind': 2,
        'wpt_vind': 3,
        'wpt_thetaind': 3
    }

    # Write YAML file with proper formatting
    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    with open(f'maps/map{iter}.yaml', 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file, Dumper=NoAliasDumper, default_flow_style=False)

    print(f"Saved map {iter}: obstacles, YAML, and CSV files.")
    print(f"Debug visualization with waypoints saved to maps/map{iter}_debug.png")


# Function removed as requested


if __name__ == '__main__':
    for i in range(NUM_MAPS):
        try:
            print(f"\nGenerating map {i}...")
            result = create_track()
            if result is None:
                print(f"Failed to create valid track for map {i}. Skipping...")
                continue
                
            track, track_int, track_ext = result

            # Generate obstacles
            obstacles = generate_obstacles(track, track_int, track_ext, args.num_obstacles)

            # Convert and save track with waypoints
            convert_track_with_waypoints(track, track_int, track_ext, i, obstacles)

        except Exception as e:
            print(f"Unexpected error during track generation for map {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    print("\nMap generation complete.")
    print("Debug visualization available at:")
    print("maps/map*_debug.png - Shows the map with waypoints as red dots")
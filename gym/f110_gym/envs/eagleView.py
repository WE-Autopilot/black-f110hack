# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from laser_models import ScanSimulator2D  # Replace 'some_module' with the actual module name

# # Initialize the scan simulator
# num_beams = 1080
# fov = 4.7
# scan_sim = ScanSimulator2D(num_beams, fov)

# # Load the map
# map_path = '/Users/main/f110hack/gym/f110_gym/envs/maps/berlin.yaml'
# map_ext = '.png'
# scan_sim.set_map(map_path, map_ext)

# # Initialize the car's pose
# pose = np.array([0.0, 0.0, 0.0])  # x, y, theta

# # Initialize the plot
# fig, ax = plt.subplots()
# ax.set_xlim(-10, 10)
# ax.set_ylim(-10, 10)
# scatter = ax.scatter([], [], s=1)

# def update(frame):
#     global pose
#     # Update the car's pose (simulate movement)
#     pose[0] += 0.1  # Move forward in x
#     pose[2] += 0.01  # Rotate slightly

#     # Generate a LIDAR scan at the current pose
#     scan = scan_sim.scan(pose, np.random.default_rng())

#     # Convert scan data to Cartesian coordinates
#     angles = np.linspace(-fov/2, fov/2, num_beams)
#     x = scan * np.cos(angles + pose[2])
#     y = scan * np.sin(angles + pose[2])

#     # Update the scatter plot
#     scatter.set_offsets(np.c_[x, y])
#     return scatter,

# # Create the animation
# ani = FuncAnimation(fig, update, frames=100, blit=True)

# # Show the plot
# plt.show()

# #ORIGINAL DO NOT DELETE

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import yaml
import os
from PIL import Image
import math

# Function to load waypoints from CSV
def load_waypoints(file_path, x_ind=0, y_ind=1, v_ind=2, theta_ind=3, delimiter=',', skip_rows=1):
    waypoints = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        # Skip header rows if needed
        for _ in range(skip_rows):
            next(reader, None)
        # Read waypoints
        for row in reader:
            if len(row) > max(x_ind, y_ind, v_ind, theta_ind):
                x = float(row[x_ind])
                y = float(row[y_ind])
                speed = float(row[v_ind])
                theta = float(row[theta_ind])
                waypoints.append([x, y, speed, theta])
    return np.array(waypoints)

# Paths
map_image_path = '/Users/main/f110hack/gym/f110_gym/envs/maps/map0.png'
waypoints_path = '/Users/main/f110hack/gym/f110_gym/envs/maps/map0.csv'
yaml_path = '/Users/main/f110hack/gym/f110_gym/envs/maps/map0.yaml'

# Load map image
try:
    map_image = np.array(Image.open(map_image_path))
    print(f"Map image loaded, shape: {map_image.shape}")
except Exception as e:
    print(f"Error loading map image: {e}")
    # Create a blank image as fallback
    map_image = np.ones((1600, 1600, 3), dtype=np.uint8) * 255
    print("Using blank image as fallback")

# Load YAML for map configuration (origin, resolution)
try:
    with open(yaml_path, 'r') as file:
        map_config = yaml.safe_load(file)
    origin_x = map_config.get('origin', [0, 0, 0])[0]
    origin_y = map_config.get('origin', [0, 0, 0])[1]
    resolution = map_config.get('resolution', 0.1) * 2 # meters per pixel
except Exception as e:
    print(f"Error loading map config: {e}")
    # Default values
    origin_x = 0
    origin_y = 0
    resolution = 0.1

# Load waypoints
try:
    waypoints = load_waypoints(
        waypoints_path,
        x_ind=int(map_config.get('wpt_xind', 0)),
        y_ind=int(map_config.get('wpt_yind', 1)),
        v_ind=int(map_config.get('wpt_vind', 2)),
        theta_ind=int(map_config.get('wpt_thetaind', 3)),
        delimiter=map_config.get('wpt_delim', ','),
        skip_rows=int(map_config.get('wpt_rowskip', 1))
    )
    
    print(f"Waypoints loaded, count: {len(waypoints)}")
    
    # Print first few waypoints for debugging
    for i in range(min(5, len(waypoints))):
        print(f"Waypoint {i}: {waypoints[i]}")
    
    # Print min/max values to check scale
    if len(waypoints) > 0:
        min_x = min(wp[0] for wp in waypoints)
        max_x = max(wp[0] for wp in waypoints)
        min_y = min(wp[1] for wp in waypoints)
        max_y = max(wp[1] for wp in waypoints)
        print(f"Waypoint bounds: X: [{min_x}, {max_x}], Y: [{min_y}, {max_y}]")
        print(f"Map origin: ({origin_x}, {origin_y}), Resolution: {resolution}")
except Exception as e:
    print(f"Error loading waypoints: {e}")
    # Sample waypoints as fallback
    waypoints_data = """x,y,speed,theta
    119.64691855978616,0.0,1.0,2.5723806128014775
    96.1383613430276,15.042124950801009,1.0,2.5723806128014775
    72.62980412626904,30.084249901602018,1.0,0.0
    72.62980412626904,30.084249901602018,1.0,2.3543549506356305
    62.01298164966187,40.74020453732835,1.0,2.35435495063563
    51.3961591730547,51.39615917305469,1.0,0.0
    51.3961591730547,51.39615917305469,1.0,1.8102297049533136"""
    
    # Parse the provided waypoints data directly
    with open("temp_waypoints.csv", "w") as f:
        f.write(waypoints_data)
    waypoints = load_waypoints("temp_waypoints.csv")
    print("Using sample waypoints as fallback")

# Function to convert world coordinates to pixel coordinates
def world_to_pixel(x, y, origin_x, origin_y, resolution, height):
    # Convert from world coordinates to pixel coordinates
    pixel_x = int((x - origin_x) / resolution)
    pixel_y = int(height - (y - origin_y) / resolution)  # Flip y since image origin is at top-left
    return pixel_x, pixel_y

# Function to extract a rotated patch around a waypoint
def get_waypoint_view(image, waypoint, origin_x, origin_y, resolution, view_size=40):
    # Convert waypoint world coordinates to pixel coordinates
    x, y, speed, theta = waypoint
    height, width = image.shape[:2]
    
    pixel_x, pixel_y = world_to_pixel(x, y, origin_x, origin_y, resolution, height)
    print(f"Waypoint {x},{y} converted to pixel {pixel_x},{pixel_y}")
    
    # Calculate the window size in pixels
    window_size_pixels = int(view_size / resolution)
    
    # Ensure the window is within the image bounds
    half_window = window_size_pixels // 2
    min_x = max(0, pixel_x - half_window)
    max_x = min(width, pixel_x + half_window)
    min_y = max(0, pixel_y - half_window)
    max_y = min(height, pixel_y + half_window)
    
    # Check if we have a valid patch
    if min_x >= max_x or min_y >= max_y:
        print(f"Warning: Invalid patch coordinates: x={min_x}:{max_x}, y={min_y}:{max_y}")
        # Create a default patch if the extracted area would be invalid
        patch = np.ones((window_size_pixels, window_size_pixels, 3), dtype=np.uint8) * 128
    else:
        # Extract the patch
        patch = image[min_y:max_y, min_x:max_x].copy()
        print(f"Extracted patch shape: {patch.shape}")
        
        # If patch is too small (near edge), pad it
        if patch.shape[0] < 10 or patch.shape[1] < 10:
            pad_y = max(0, 10 - patch.shape[0])
            pad_x = max(0, 10 - patch.shape[1])
            if len(patch.shape) == 3:  # Color image
                patch = np.pad(patch, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant', constant_values=128)
            else:  # Grayscale
                patch = np.pad(patch, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=128)
    
    # Use PIL for rotation to avoid interpolation issues
    from PIL import Image
    
    # Ensure patch is not empty
    if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2:
        print("Warning: Empty patch detected, creating default patch")
        patch = np.ones((window_size_pixels, window_size_pixels, 3), dtype=np.uint8) * 128
    
    # Convert to PIL for rotation
    try:
        patch_pil = Image.fromarray(patch)
        # Adjust theta by -π/2 to make 0 radians point upward
        rotation_angle = np.degrees(theta - np.pi/2)
        rotated_patch = np.array(patch_pil.rotate(rotation_angle, expand=True))
        print(f"Rotated patch shape: {rotated_patch.shape}")
    except Exception as e:
        print(f"Error during rotation: {e}")
        # Return original patch if rotation fails
        rotated_patch = patch
    
    return rotated_patch, (pixel_x, pixel_y)

# Create figure and axis for visualization
fig, ax = plt.subplots(figsize=(15, 8))  # Increased figure size

# Current waypoint index
current_waypoint_idx = 0

# Plot the waypoints on the full map
height, width = map_image.shape[:2]
full_map_ax = plt.subplot(121)
full_map_ax.imshow(map_image)  # Display the map without adjusting extent
waypoint_pixels = [world_to_pixel(wp[0], wp[1], origin_x, origin_y, resolution, height) for wp in waypoints]
full_map_ax.plot([p[0] for p in waypoint_pixels], [p[1] for p in waypoint_pixels], 'g-', alpha=0.7)
full_map_ax.scatter([p[0] for p in waypoint_pixels], [p[1] for p in waypoint_pixels], c='g', s=30)
full_map_ax.set_title('Full Track with Waypoints')

# Create the waypoint view
waypoint_view_ax = plt.subplot(122)
waypoint_marker = waypoint_view_ax.plot([], [], 'ro', markersize=10)[0]
direction_line = waypoint_view_ax.plot([], [], 'r-', linewidth=2)[0]

# Function to update the visualization for each waypoint
def update(frame):
    global current_waypoint_idx
    
    # Get current waypoint
    current_waypoint_idx = frame % len(waypoints)
    current_waypoint = waypoints[current_waypoint_idx]
    
    # Extract and rotate the view around the waypoint
    rotated_patch, (pixel_x, pixel_y) = get_waypoint_view(
        map_image, current_waypoint, origin_x, origin_y, resolution, view_size=40
    )
    
    # Update the waypoint view
    waypoint_view_ax.clear()
    waypoint_view_ax.imshow(rotated_patch)
    waypoint_view_ax.set_title(f'Waypoint {current_waypoint_idx}: θ = {current_waypoint[3]:.2f} rad')
    
    # Highlight the current waypoint on the full map
    full_map_ax.clear()
    full_map_ax.imshow(map_image)  # Just display the image without adjusting extent
    full_map_ax.plot([p[0] for p in waypoint_pixels], [p[1] for p in waypoint_pixels], 'g-', alpha=0.7)
    full_map_ax.scatter([p[0] for p in waypoint_pixels], [p[1] for p in waypoint_pixels], c='g', s=20)
    full_map_ax.scatter(pixel_x, pixel_y, c='r', s=100)
    full_map_ax.set_title('Full Track with Current Waypoint')
    
    # Add a box around the current view area
    half_window = int(20 / resolution)
    rect = plt.Rectangle(
        (pixel_x - half_window, pixel_y - half_window),
        2 * half_window, 2 * half_window,
        linewidth=2, edgecolor='r', facecolor='none'
    )
    full_map_ax.add_patch(rect)
    
    return waypoint_view_ax, full_map_ax

# Create the animation
ani = FuncAnimation(fig, update, frames=len(waypoints), interval=1000, blit=False)

plt.tight_layout()
plt.show()

# this is pretty close. for some reason parts of the map aren't showing. what value are the waypoints mapping to actually?
# i think the issue is that the map is too big and the waypoints are too small. i need to scale the map down and then scale the waypoints up
# i keep getting warnings like this: Waypoint 63.50131827497712,-63.50131827497715 converted to pixel 515,2235
# Warning: Invalid patch coordinates: x=315:715, y=2035:1600
# Rotated patch shape: (400, 400, 3)
# Waypoint 63.50131827497712,-63.50131827497715 converted to pixel 515,2235
# Warning: Invalid patch coordinates: x=315:715, y=2035:1600
# Rotated patch shape: (504, 504, 3)
# Waypoint 88.93858997243024,-55.438675693375984 converted to pixel 769,2154
# Warning: Invalid patch coordinates: x=569:969, y=1954:1600







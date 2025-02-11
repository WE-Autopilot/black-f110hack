import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from laser_models import ScanSimulator2D  # Replace 'laser_models' with the actual module name

# Initialize the scan simulator
num_beams = 1080  # Number of LiDAR beams
fov = 2 * np.pi  # Full 360-degree field of view
scan_sim = ScanSimulator2D(num_beams, fov)

# Load the map
map_path = '/Users/main/f110hack/gym/f110_gym/envs/maps/berlin.yaml'
map_ext = '.png'
scan_sim.set_map(map_path, map_ext)

# Initialize the car's pose
pose = np.array([0.0, 0.0, 0.0])  # x, y, theta

# Initialize the plot
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
scatter = ax.scatter([], [], s=1, label='LiDAR Points')
ax.scatter(0, 0, marker='x', color='red', s=100, label='Car Position')  # Car position
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_title('360-Degree LiDAR View')
ax.grid(True)
ax.axis('equal')  # Ensure the plot is scaled equally
ax.legend()

def update(frame):
    global pose
    # Update the car's pose (simulate movement)
    pose[0] += 0.1  # Move forward in x
    pose[2] += 0.01  # Rotate slightly

    # Generate a LiDAR scan at the current pose
    scan = scan_sim.scan(pose, np.random.default_rng())

    # Convert scan data to Cartesian coordinates
    angles = np.linspace(-fov/2, fov/2, num_beams)  # Full 360-degree angles
    x = scan * np.cos(angles + pose[2])
    y = scan * np.sin(angles + pose[2])

    # Update the scatter plot
    scatter.set_offsets(np.c_[x, y])
    return scatter,

# Create the animation
ani = FuncAnimation(fig, update, frames=100, blit=True, interval=50)

# Show the plot
plt.show()
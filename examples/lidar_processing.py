# lidar_processing.py
import numpy as np
import matplotlib.pyplot as plt

def process_and_visualize_lidar(lidar_scan, pose_theta):
    """
    Process LiDAR data and visualize it in a 360-degree view.
    
    Args:
        lidar_scan (np.array): Array of LiDAR scan distances.
        pose_theta (float): Current orientation of the car (in radians).
    """
    # Generate angles for a full 360-degree view
    angles = np.linspace(-np.pi, np.pi, len(lidar_scan))
    
    # Convert LiDAR scan to Cartesian coordinates
    x = lidar_scan * np.cos(angles + pose_theta)
    y = lidar_scan * np.sin(angles + pose_theta)
    
    # Plot the LiDAR data
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=1, label='LiDAR Points')
    plt.scatter(0, 0, marker='x', color='red', s=100, label='Car Position')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('360-Degree LiDAR View')
    plt.grid(True)
    plt.axis('equal')  # Ensure the plot is scaled equally
    plt.legend()
    plt.show()
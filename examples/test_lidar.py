# test_lidar.py
import numpy as np
from lidar_processing import process_and_visualize_lidar

def generate_sample_lidar_data():
    """
    Generate synthetic LiDAR data for testing.
    """
    # Simulate a LiDAR scan with 360 points (one per degree)
    num_points = 360
    angles = np.linspace(-np.pi, np.pi, num_points)
    
    # Simulate distances (e.g., a circular wall at 5 meters)
    distances = np.ones(num_points) * 5.0
    
    # Add some noise to simulate real-world data
    noise = np.random.normal(0, 0.1, num_points)
    distances += noise
    
    return distances

if __name__ == '__main__':
    # Generate sample LiDAR data
    lidar_scan = generate_sample_lidar_data()
    
    # Simulate car orientation (e.g., 0 radians means facing right)
    pose_theta = 0.0
    
    # Process and visualize the LiDAR data
    process_and_visualize_lidar(lidar_scan, pose_theta)
import numpy as np
from scipy.interpolate import interp1d
import cv2

def make_360_scan(scan, fov=360):
    """
    Convert a LiDAR scan with a given FOV to a full 360-degree scan.
    
    :param scan: 1D array of LiDAR distances
    :param fov: Field of View of the LiDAR in degrees (e.g., 270 for a typical LiDAR)
    :return: 1D array representing a 360-degree LiDAR scan
    """
    num_points = len(scan)
    # Convert angles to be centered around 0
    angles = np.linspace(-fov/2, fov/2, num_points)
    
    # Create a full 360-degree angle array with the same density of points
    points_per_degree = num_points / fov
    num_full_points = int(360 * points_per_degree)
    full_angles = np.linspace(-180, 180, num_full_points)
    
    # Interpolate the scan data to fill in the missing angles
    interp_fn = interp1d(angles, scan, kind='linear', bounds_error=False, fill_value=np.max(scan))
    full_scan = interp_fn(full_angles)
    
    return full_scan

def lidar_to_image(scan, img_size=256):
    """
    Convert a 360-degree LiDAR scan to a 256x256 image.
    
    :param scan: 1D array of LiDAR distances (360 degrees)
    :param img_size: Size of the output image (default: 256x256)
    :return: 2D numpy array representing the LiDAR scan as an image
    """
    # Create angle array matching the scan length
    angles = np.linspace(-180, 180, len(scan))
    
    # Convert polar coordinates to Cartesian
    x = scan * np.cos(np.radians(angles))
    y = scan * np.sin(np.radians(angles))
    
    # Create an empty image with a white background
    image = np.ones((img_size, img_size), dtype=np.uint8) * 255
    
    # Find the scale to fit points in image
    # Use a margin to prevent points from being at the very edge
    margin = 0.1
    max_range = np.max(np.abs([x, y]))
    scale = (img_size * (1 - margin)) / (2 * max_range)
    
    # Scale and center the points
    x_scaled = (x * scale + img_size/2).astype(int)
    y_scaled = (y * scale + img_size/2).astype(int)
    
    # Clip points to image boundaries
    mask = (x_scaled >= 0) & (x_scaled < img_size) & (y_scaled >= 0) & (y_scaled < img_size)
    x_scaled = x_scaled[mask]
    y_scaled = y_scaled[mask]
    
    # Draw the points and connect them
    for i in range(len(x_scaled)):
        # Draw points as small filled circles
        cv2.circle(image, (x_scaled[i], y_scaled[i]), 2, 0, -1)
        
        # Connect points with lines
        if i > 0:
            cv2.line(image, 
                    (x_scaled[i-1], y_scaled[i-1]),
                    (x_scaled[i], y_scaled[i]),
                    0, 1)
    
    # Draw the car position (center)
    cv2.circle(image, (img_size//2, img_size//2), 4, 128, -1)
    
    return image
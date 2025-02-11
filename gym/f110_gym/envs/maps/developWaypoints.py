import numpy as np
import cv2
from scipy.interpolate import interp1d
import csv
import argparse

def generate_waypoints(track_coordinates, num_waypoints=100, max_angle_change=30):
    """
    Generate a set of waypoints for a racetrack map.
    
    Parameters:
    track_coordinates (np.array): 2D array of (x, y) coordinates representing the track.
    num_waypoints (int): Number of waypoints to generate.
    max_angle_change (float): Maximum allowable change in angle between consecutive waypoints (in degrees).
    
    Returns:
    np.array: 3D array of (x, y, theta) waypoints.
    """
    # Interpolate the track coordinates to get a smooth, continuous track representation
    x = track_coordinates[:, 0]
    y = track_coordinates[:, 1]
    track_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    s = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)) / track_length
    f_x = interp1d(s, x)
    f_y = interp1d(s, y)

    # Generate waypoints along the track
    s_waypoints = np.linspace(0, 1, num_waypoints)
    x_waypoints = f_x(s_waypoints)
    y_waypoints = f_y(s_waypoints)

    # Calculate the angle at each waypoint
    theta_waypoints = []
    for i in range(len(x_waypoints)):
        if i == 0:
            theta = np.arctan2(y_waypoints[1] - y_waypoints[0], x_waypoints[1] - x_waypoints[0])
        else:
            theta = np.arctan2(y_waypoints[i] - y_waypoints[i-1], x_waypoints[i] - x_waypoints[i-1])
        theta_waypoints.append(theta)

    # Ensure the angle change between consecutive waypoints is within the specified limit
    for i in range(1, len(theta_waypoints)):
        angle_change = np.abs(np.degrees(theta_waypoints[i] - theta_waypoints[i-1]))
        if angle_change > max_angle_change:
            theta_waypoints[i] = theta_waypoints[i-1] + np.radians(max_angle_change) * np.sign(theta_waypoints[i] - theta_waypoints[i-1])

    return np.column_stack((x_waypoints, y_waypoints, theta_waypoints))

def generate_and_save_waypoints(track_coordinates, num_waypoints=100, max_angle_change=30, output_file='berlin.csv'):
    """
    Generate a set of waypoints for a racetrack map and save them to a CSV file.
    
    Parameters:
    track_coordinates (np.array): 2D array of (x, y) coordinates representing the track.
    num_waypoints (int): Number of waypoints to generate.
    max_angle_change (float): Maximum allowable change in angle between consecutive waypoints (in degrees).
    output_file (str): Path to the output CSV file.
    """
    # Generate the waypoints
    waypoints = generate_waypoints(track_coordinates, num_waypoints, max_angle_change)
    
    # Save the waypoints to a CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerows(waypoints)
    
    print(f'Waypoints saved to: {output_file}')

def generate_and_save_waypoints_from_image(image_path = 'berlin.png', num_waypoints=100, max_angle_change=30, output_file='berlin.csv'):
    """
    Generate a set of waypoints for a racetrack map from an image and save them to a CSV file.
    
    Parameters:
    image_path (str): Path to the image of the racetrack.
    num_waypoints (int): Number of waypoints to generate.
    max_angle_change (float): Maximum allowable change in angle between consecutive waypoints (in degrees).
    output_file (str): Path to the output CSV file.
    """
    # Load the racetrack image
    image = cv2.imread(image_path)
    
    # Detect the track boundaries using edge detection and contour extraction
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour, which should represent the track
    track_contour = max(contours, key=cv2.contourArea)
    
    # Extract the (x, y) coordinates of the track contour
    track_coordinates = np.squeeze(track_contour, axis=1)
    
    # Ensure the x and y arrays have the same length
    x = track_coordinates[:, 0]
    y = track_coordinates[:, 1]
    if len(x) != len(y):
        track_coordinates = track_coordinates[:min(len(x), len(y))]
        x = track_coordinates[:, 0]
        y = track_coordinates[:, 1]
    
    # Generate the waypoints and save them to a CSV file
    generate_and_save_waypoints(track_coordinates, num_waypoints, max_angle_change, output_file)

def main():
    parser = argparse.ArgumentParser(description='Generate waypoints from a racetrack image.')
    parser.add_argument('--image_path', type=str, default='berlin.png', help='Path to the racetrack image')
    parser.add_argument('--num_waypoints', type=int, default=100, help='Number of waypoints to generate')
    parser.add_argument('--max_angle_change', type=float, default=30, help='Maximum allowable change in angle between consecutive waypoints (in degrees)')
    parser.add_argument('--output_file', type=str, default='berlin.csv', help='Path to the output CSV file')
    
    args = parser.parse_args()
    
    generate_and_save_waypoints_from_image(args.image_path, args.num_waypoints, args.max_angle_change, args.output_file)

if __name__ == '__main__':
    main()
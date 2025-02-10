import numpy as np
from PIL import Image

def get_overlap_distance(image, position, theta, path):
    """
    Calculate the total distance traveled out of bounds.

    :param image: PIL image (map) where obstacles are identified.
    :param position: (x, y) NumPy array representing the car's initial position.
    :param theta: Float representing the car's orientation (rotation in radians).
    :param path: NumPy array of shape (128,) representing the movement sequence (reshaped into 64 2D vectors).
    :return: Float representing the total out-of-bounds distance.
    """

    # Reshape path into 64 (x, y) vectors
    vectors = path.reshape(64, 2)

    # Create rotation matrix
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta),  np.cos(theta)]])

    # Apply rotation & translation (vectorized)
    transformed_path = position + np.dot(vectors, rotation_matrix.T)

    # Convert image to grayscale for boundary checking
    image = image.convert("L")
    image_array = np.array(image)
    height, width = image_array.shape

    # Define in-bounds threshold (assuming white pixels are valid paths)
    in_bounds_threshold = 200  # Adjust based on map color scheme

    # Extract x and y coordinates
    x, y = transformed_path[:, 0], transformed_path[:, 1]
    
    # Convert to integer indices for image lookup
    x_int, y_int = x.astype(int), y.astype(int)

    # Out-of-bounds mask
    out_of_bounds_mask = (x_int < 0) | (x_int >= width) | (y_int < 0) | (y_int >= height)

    # Obstacle mask
    obstacle_mask = image_array[y_int, x_int] < in_bounds_threshold

    # Combine invalid positions
    invalid_mask = out_of_bounds_mask | obstacle_mask

    # Compute out-of-bounds distance
    out_of_bounds_distance = np.sum(np.linalg.norm(np.diff(transformed_path[invalid_mask], axis=0), axis=1))

    return out_of_bounds_distance

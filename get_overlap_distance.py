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
    
    # Apply rotation and translation
    transformed_path = np.array([position + rotation_matrix @ vector for vector in vectors])
    
    # Convert image to grayscale for boundary checking
    image = image.convert("L")
    image_array = np.array(image)
    height, width = image_array.shape
    
    # Define in-bounds threshold (assuming white pixels are valid paths)
    in_bounds_threshold = 200  # Adjust based on map color scheme
    
    # Calculate total out-of-bounds distance
    out_of_bounds_distance = 0.0
    
    for i in range(len(transformed_path) - 1):
        x1, y1 = transformed_path[i]
        x2, y2 = transformed_path[i + 1]
        
        # Check if points are within the image boundaries
        if not (0 <= int(x1) < width and 0 <= int(y1) < height):
            out_of_bounds_distance += np.linalg.norm([x2 - x1, y2 - y1])
            continue
        
        if not (0 <= int(x2) < width and 0 <= int(y2) < height):
            out_of_bounds_distance += np.linalg.norm([x2 - x1, y2 - y1])
            continue
        
        # Check if the segment is out of bounds (i.e., crosses dark/obstacle areas)
        if image_array[int(y1), int(x1)] < in_bounds_threshold or image_array[int(y2), int(x2)] < in_bounds_threshold:
            out_of_bounds_distance += np.linalg.norm([x2 - x1, y2 - y1])
    
    return out_of_bounds_distance

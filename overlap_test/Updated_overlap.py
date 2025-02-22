import numpy as np
from PIL import Image, ImageDraw

def get_absolute_path(path, position, angle, scale):
    """
    Converts relative movements into absolute positions with scaling and rotation.
    """
    path = path.reshape(-1, 2) * scale
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    path = (rotation_matrix @ path.T).T
    absolute_path = np.vstack([np.zeros((1, 2)), np.cumsum(path, axis=0)]) + position
    return absolute_path

def get_overlap(track_image, path, position, angle, scale, car_radius):
    """
    Computes the overlap area by masking the path on an inverted, normalized map.
    :param track_image: PIL image of the map
    :param path: Array of relative path movements
    :param position: Initial position of the car
    :param angle: Rotation angle in radians
    :param scale: Scaling factor
    :param car_radius: Radius of the car to determine path thickness
    :return: Overlap area (sum of out-of-bounds pixels)
    """
    track_array = np.array(track_image.convert("L"))
    absolute_path = get_absolute_path(path, position, angle, scale)
    
    # Create a blank black image (same size as track)
    path_only_image = Image.new("L", (track_array.shape[1], track_array.shape[0]), 0)
    draw = ImageDraw.Draw(path_only_image)
    
    # Draw the path in white (255) with a thickness based on car_radius
    for i in range(len(absolute_path) - 1):
        x1, y1 = absolute_path[i]
        x2, y2 = absolute_path[i + 1]
        draw.line([(x1, y1), (x2, y2)], fill=255, width=car_radius * 2)
    
    # Invert and normalize the track image
    inverted_track_array = 255 - track_array
    normalized_track_array = inverted_track_array / 255.0
    
    # Convert path-only image to an array and normalize
    path_mask_array = np.array(path_only_image) / 255.0
    
    # Apply the mask: Keep only the path pixels on the inverted map
    masked_map = normalized_track_array * path_mask_array
    
    # Ensure the masked map is correctly normalized
    assert masked_map.min() >= 0 and masked_map.max() <= 1, "Masked map is not normalized!"
    
    # Compute overlap area
    overlap_area = np.sum(masked_map)
    return overlap_area

import cv2
import numpy as np

def calculate_out_of_bounds_distance(image_path, p1, p2):
    """
    Calculate the length of the line segment between p1 and p2 that is inside a wall.
    
    Parameters:
        image_path (str): Path to the black-and-white map image.
        p1 (tuple): (x1, y1) coordinates of the first point.
        p2 (tuple): (x2, y2) coordinates of the second point.
    
    Returns:
        float: The length of the segment inside the walls.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert to binary: walls are black (0), free space is white (255)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Generate points along the line using Bresenham's algorithm
    x1, y1 = p1
    x2, y2 = p2

    # Bresenham's line algorithm to get discrete points along the line
    line_pixels = []
    steep = abs(y2 - y1) > abs(x2 - x1)
    
    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    dx = x2 - x1
    dy = abs(y2 - y1)
    error = dx // 2
    ystep = 1 if y1 < y2 else -1

    y = y1
    for x in range(x1, x2 + 1):
        coord = (y, x) if steep else (x, y)
        line_pixels.append(coord)
        error -= dy
        if error < 0:
            y += ystep
            error += dx

    if swapped:
        line_pixels.reverse()

    # Count the number of pixels that are inside a wall
    in_wall_count = sum(1 for x, y in line_pixels if binary_image[y, x] == 255)

    # Convert pixel count to distance assuming 1 pixel = 1 unit length
    out_of_bounds_length = in_wall_count

    return out_of_bounds_length

# Example usage with the provided image
image_path = "/mnt/data/image.png"
p1 = (50, 50)  # Example start point
p2 = (150, 150)  # Example end point

out_of_bounds_distance = calculate_out_of_bounds_distance(image_path, p1, p2)
out_of_bounds_distance

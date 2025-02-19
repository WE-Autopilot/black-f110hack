import numpy as np

def get_overlap_distance_2(image, position, theta, path):
    """
    Compute total out-of-bounds distance for a path on a given map.

    Args:
        image (PIL.Image): map where black pixels are out of bounds
        position (np.array): car's (x, y) in world coords
        theta (float): car heading in radians
        path (np.array): sequence of (dx, dy) steps

    Returns:
        float: Total traveled distance that is out of bounds
    """

    # Crop to 256x256
    image = image.crop((0, 0, 256, 256))
    img_array = np.array(image)
    cx, cy = 128, 128  # image center

    def world_to_image(pt):
        # Translate so 'position' -> (0,0), rotate by -theta, then offset to centre
        dx = pt[0] - position[0]
        dy = pt[1] - position[1]
        c = np.cos(-theta)
        s = np.sin(-theta)
        rx = dx * c - dy * s
        ry = dx * s + dy * c
        return (cx + rx, cy + ry)

    def is_out_of_bounds(img, x, y):
        h, w = img.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return True
        pixel = img[y, x]
        if pixel.ndim == 0: # grayscale
            return (pixel < 10)
        else: # RGB
            return (pixel.sum() < 30)

    # Build absolute positions along path
    positions = [position.copy()]
    for step in path:
        positions.append(positions[-1] + step)

    # Accumulate out-of-bounds distance
    total_out = 0.0
    for i in range(len(positions) - 1):
        p0, p1 = positions[i], positions[i + 1]
        seg_len = np.linalg.norm(p1 - p0)
        x_img, y_img = world_to_image(p1)
        x_img, y_img = int(round(x_img)), int(round(y_img))
        if is_out_of_bounds(img_array, x_img, y_img):
            total_out += seg_len

    return float(total_out)
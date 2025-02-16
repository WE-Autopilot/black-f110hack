import numpy as np
from PIL import Image, ImageDraw
from stage1_penalty import stage1_penalty

def draw_x(draw, center, size=5, color='red'):
    """ Draws an 'X' at the given center position. """
    x, y = center
    draw.line((x - size, y - size, x + size, y + size), fill=color, width=2)
    draw.line((x - size, y + size, x + size, y - size), fill=color, width=2)

def make_realistic_map(target, position, path):
    """
    Creates a 256x256 map with:
      - White in-bounds area
      - Black square obstacles
      - A red 'X' marking the target
      - A realistic path drawn in blue
    """
    img = Image.new('RGB', (256, 256), 'black')
    d = ImageDraw.Draw(img)

    # White drivable area
    d.rectangle([20, 20, 236, 236], fill='white')

    # Black square obstacles
    d.rectangle([50, 50, 100, 100], fill='black')  # Upper-left
    d.rectangle([140, 40, 190, 90], fill='black')  # Upper-right
    d.rectangle([80, 130, 130, 180], fill='black')  # Middle
    d.rectangle([150, 160, 210, 220], fill='black')  # Bottom-right

    # Draw target with an 'X'
    draw_x(d, target, size=6, color='red')

    # Compute absolute positions for path
    absolute_path = [position]
    for step in path:
        absolute_path.append(absolute_path[-1] + step)

    # Convert absolute positions to integer coordinates
    path_coords = [(int(x), int(y)) for x, y in absolute_path]

    # Draw path as blue line
    d.line(path_coords, fill="blue", width=2)

    # Draw path points as small blue circles
    for point in path_coords:
        d.ellipse([point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2], fill="blue")

    return img

def main():
    # Define target (goal location)
    target = np.array([210, 50])  # Positioned towards upper-right

    # Define car's start position
    position = np.array([40.0, 220.0])  # Start at bottom-left

    # Define a more realistic path that curves around obstacles
    path = np.array([
        [10, -10], [10, -10], [10, -5],   # Move diagonally up-right
        [5, -10],  [5, -10],              # Move more upwards
        [10, -5],  [10, -5], [10, 0],     # Slight right, avoiding middle obstacle
        [5, 5],    [5, 5], [5, 5],        # Curving up right
        [10, -10], [10, -10], [10, -5],   # Move towards target
    ])

    # Generate and show map with target and path
    map_image = make_realistic_map(target, position, path)
    map_image.show()

    # Weights for penalty components
    a, b, c, d = 1.0, 1.0, 0.1, 1.0

    # Compute penalty
    penalty = stage1_penalty(map_image, position, 0.0, path, target, a, b, c, d)
    print("Stage 1 Penalty =", penalty)

if __name__ == "__main__":
    main()

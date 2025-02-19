import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Given parameters
path = np.array([
    [-5, -3], [-8, -8], [-8, -2], [-4, -2],
    [-9, 1], [-6, -1], [-8, 2], [-7, -1],
    [-2, -5], [-3, -5], [-1, -6], [1, -6],
    [2, -6], [-1, -6], [-1, -6], [1, -12]
])
position = np.array([452, 224])
angle = 0  # No rotation needed

# Convert relative movements to absolute positions (tip-to-tail)
absolute_path = [position]
for vector in path:
    absolute_path.append(absolute_path[-1] + vector)
absolute_path = np.array(absolute_path)

# Load the track image (assuming last uploaded file is the map)
image_path = "/mnt/data/image.png"
track_image = Image.open(image_path).convert("L")
track_array = np.array(track_image)

# Plot the track with the path overlay
plt.figure(figsize=(8, 6))
plt.imshow(track_array, cmap="gray", origin="upper")
plt.plot(absolute_path[:, 0], absolute_path[:, 1], color="blue", linewidth=2)
plt.title("Path Overlaid on Track")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.show()

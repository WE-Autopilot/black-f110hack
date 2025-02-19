import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Given parameters
path = np.array([
    [-5, -3], [-8, -8], [-8, -2], [-4, -2],
    [-9, 1], [-6, -1], [-8, 2], [-7, -1],
    [-2, -5], [-3, -5], [-1, -6], [1, -6],
    [2, -6], [-1, -6], [-1, -6], [1, -12]
])
position = np.array([452, 224])
angle = 0  # No rotation needed

# Convert relative movements to absolute positions (tip-to-tail accumulation)
absolute_path = [position]
for vector in path:
    absolute_path.append(absolute_path[-1] + vector)
absolute_path = np.array(absolute_path)

# Load the track image (assuming last uploaded file is the map)
image_path = "exmap.png"
track_image = Image.open(image_path).convert("L")
track_array = np.array(track_image)

# -------- Step 1: Generate Path-Only Image --------
# Create a blank black image (same size as track)
path_only_image = Image.new("L", (track_array.shape[1], track_array.shape[0]), 0)
draw = ImageDraw.Draw(path_only_image)

# Draw the path in white (255)
for i in range(len(absolute_path) - 1):
    x1, y1 = absolute_path[i]
    x2, y2 = absolute_path[i + 1]
    draw.line([(x1, y1), (x2, y2)], fill=255, width=2)

# Save the path-only image
path_only_image_path = "/mnt/data/path_only.png"
path_only_image.save(path_only_image_path)
print(f"Path-only image saved: {path_only_image_path}")

# -------- Step 2: Plot Path Overlaid on Track --------
plt.figure(figsize=(8, 6))
plt.imshow(track_array, cmap="gray", origin="upper")
plt.plot(absolute_path[:, 0], absolute_path[:, 1], color="blue", linewidth=2)
plt.title("Path Overlaid on Track")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.show()

# -------- Step 3: Plot and Display Path-Only Image --------
plt.figure(figsize=(8, 6))
plt.imshow(path_only_image, cmap="gray", origin="upper")
plt.title("Path-Only Image")
plt.axis("off")  # Hide axes for a clean image
plt.show()

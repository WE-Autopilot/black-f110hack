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
car_radius = 3  # Determines path thickness

# Convert relative movements to absolute positions (tip-to-tail accumulation)
absolute_path = [position]
for vector in path:
    absolute_path.append(absolute_path[-1] + vector)
absolute_path = np.array(absolute_path)

# Load the track image
image_path = "exmap.png"
track_image = Image.open(image_path).convert("L")
track_array = np.array(track_image)

# -------- Step 1: Generate Path-Only Image --------
# Create a blank black image (same size as track)
path_only_image = Image.new("L", (track_array.shape[1], track_array.shape[0]), 0)
draw = ImageDraw.Draw(path_only_image)

# Draw the path in white (255) with a thickness based on car_radius
for i in range(len(absolute_path) - 1):
    x1, y1 = absolute_path[i]
    x2, y2 = absolute_path[i + 1]
    draw.line([(x1, y1), (x2, y2)], fill=255, width=car_radius * 2)

# Save the path-only image
path_only_image_path = "path_only_with_radius.png"
path_only_image.save(path_only_image_path)
print(f"Path-only image saved: {path_only_image_path}")

# -------- Step 2: Invert and Normalize the Map Image --------
# Invert the grayscale map (track becomes black, out-of-bounds becomes white)
inverted_track_array = 255 - track_array

# Normalize the inverted map image (scale pixel values between 0 and 1)
normalized_track_array = inverted_track_array / 255.0

# -------- Step 3: Apply Path Mask to Map --------
# Convert path-only image to an array and normalize
path_mask_array = np.array(path_only_image) / 255.0

# Apply the mask: Keep only the path pixels on the inverted map
masked_map = normalized_track_array * path_mask_array

# -------- Step 4: Save and Display Results --------
# Save the masked image
masked_map_path = "masked_map.png"
plt.imsave(masked_map_path, masked_map, cmap="gray")
print(f"Masked map image saved: {masked_map_path}")

# Show the path-only image
plt.figure(figsize=(8, 6))
plt.imshow(path_only_image, cmap="gray", origin="upper")
plt.title(f"Path-Only Image (Radius = {car_radius})")
plt.axis("off")
plt.show()

# Show the masked map image
plt.figure(figsize=(8, 6))
plt.imshow(masked_map, cmap="gray", origin="upper")
plt.title("Masked Map (Track Inverted & Path Applied)")
plt.axis("off")
plt.show()

# -------- Step 5: Sum the Normalized Pixel Values --------
# Ensure the masked map is correctly normalized (values between 0 and 1)
assert masked_map.min() >= 0 and masked_map.max() <= 1, "Masked map is not normalized!"

# Sum all pixel values along the single channel (since it's a grayscale image)
total_out_of_bounds_distance = np.sum(masked_map)

# Display the result
print(f"Total Out-of-Bounds Distance (Summed Pixel Values): {total_out_of_bounds_distance}")

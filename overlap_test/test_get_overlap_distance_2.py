import numpy as np
from PIL import Image, ImageDraw
from get_overlap_distance_2 import get_overlap_distance_2

def make_realistic_test_image():
    # Create 256x256 black canvas
    img = Image.new('RGB', (256, 256), 'black')
    d = ImageDraw.Draw(img)

    # Draw large white rectangle as "in-bounds" lane
    d.rectangle([20, 20, 236, 236], fill='white')

    # Draw a black diagonal rectangle (obstacle) in middle
    # rotate a small rectangle around its center by 30 degrees
    cx, cy = 140, 140
    w, h = 40, 80  # obstacle width and height
    angle = np.radians(30)

    def rotate(px, py, ox, oy, a):
        # rotate point (px,py) around (ox,oy) by angle a
        dx, dy = px - ox, py - oy
        rx = dx*np.cos(a) - dy*np.sin(a)
        ry = dx*np.sin(a) + dy*np.cos(a)
        return (ox + rx, oy + ry)

    # corners of the rectangle around (cx,cy)
    corners = [
        (cx - w/2, cy - h/2),
        (cx + w/2, cy - h/2),
        (cx + w/2, cy + h/2),
        (cx - w/2, cy + h/2),
    ]
    # rotate each corner by 'angle'
    rot_corners = [rotate(x, y, cx, cy, angle) for (x, y) in corners]
    d.polygon(rot_corners, fill='black')
    return img

def main():
    image = make_realistic_test_image()
    image.show()  # See the lane + obstacle

    # Start position (world coords)
    position = np.array([60.0, 40.0])
    theta = 0.0  # no rotation

    # A path that arcs to the right and may partially cross the black obstacle
    path = np.array([
        [0, 15],
        [8,  8],
        [8,  0],
        [8, -5],
        [8, -3],
        [8, -2],
    ])

    dist_out = get_overlap_distance_2(image, position, theta, path)
    print("Distance out of bounds:", dist_out)

if __name__ == "__main__":
    main()

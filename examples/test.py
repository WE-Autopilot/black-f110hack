import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml

image = Image.open('example_map.png')
with open('example_map.yaml', 'r') as file:
    config = yaml.safe_load(file)
waypoints = np.loadtxt('example_waypoints.csv', delimiter=';', skiprows=3)[:, 1:3]


origin = np.array(config["origin"])[:2]

waypoints -= origin
waypoints /= config["resolution"]
waypoints[:, 1] = image.height - waypoints[:, 1]


plt.imshow(image)
plt.scatter(*waypoints.T)
plt.show()

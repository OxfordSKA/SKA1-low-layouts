"""Module to generate regular station layouts for SKA1-Low v4a."""
import math
import matplotlib.pyplot as plt
import numpy
from os.path import join


def rotate_station(theta, coordinates):
    """ Rotate station coordinates.

    Args:
        theta (float): Rotation angle, in radians.
        coordinates (array_like): Station coordinates.

    Returns:
        rotated station coordinates.
    """
    rot = numpy.array([[math.cos(theta), -math.sin(theta), 0.0],
                       [math.sin(theta), math.cos(theta), 0.0],
                       [0.0, 0.0, 1.0]])
    return numpy.dot(coordinates, rot.T)


telescope = 'SKA1_low_v4a_regular.tm'
layout_file_enu = join(telescope, 'layout_enu.txt')
layout_enu = numpy.loadtxt(layout_file_enu)
station_file = 'sbfile3r0.ant'
station_coords = numpy.loadtxt(station_file)

station_layout = dict()

# station 0
station_layout[0] = numpy.copy(station_coords)

# station 1
s = 1
station_layout[s] = numpy.copy(station_coords)
cx = layout_enu[s, 0]
cy = layout_enu[s, 1]
station_layout[s] = rotate_station(-math.atan(cx / cy), station_layout[1])
station_layout[s][:, 0] += cx
station_layout[s][:, 1] += cy


# Plot the layout
fig = plt.figure(figsize=(6.5, 6.5))
ax = fig.add_subplot(111, aspect='equal')
x = layout_enu[:, 0]
y = layout_enu[:, 1]
ax.plot(x, y, 'k+')
ax.grid(True)
ax.set_xlim((-30, 10))
ax.set_ylim((-30, 10))
x = station_layout[0][:, 0]
y = station_layout[0][:, 1]
ax.plot(x, y, 'r+')
x = station_layout[1][:, 0]
y = station_layout[1][:, 1]
ax.plot(x, y, 'g+')
plt.show()

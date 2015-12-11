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
coords_3r0 = numpy.loadtxt('sbfile3r0.ant')
coords_4r0 = numpy.loadtxt('sbfile4r0.ant')

# Plot the layout
fig = plt.figure(figsize=(6.5, 6.5))
ax = fig.add_subplot(111, aspect='equal')
# Plot station positions
x = layout_enu[:, 0]
y = layout_enu[:, 1]
ax.plot(x, y, 'k+')
# Plot station 3
ax.grid(True)
x = coords_3r0[:, 0] + layout_enu[0, 0]
y = coords_3r0[:, 1] + layout_enu[0, 1]
ax.plot(x, y, 'r.')
# Plot station 4
x = coords_4r0[:, 0] + layout_enu[3, 0]
y = coords_4r0[:, 1] + layout_enu[3, 1]
ax.plot(x, y, 'g.')
# ax.set_xlim(-60, 60)
# ax.set_ylim(-60, 60)
plt.show()

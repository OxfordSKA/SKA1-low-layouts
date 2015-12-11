"""Module to test rotation of stations."""
import math
import matplotlib.pyplot as plt
import numpy


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
    return numpy.dot(coordinates, rot)


coords_3r0 = numpy.loadtxt('sbfile3r0.ant')
station_file = 'sbfile4r0.ant'
coords_4r0 = numpy.loadtxt('sbfile4r0.ant')

# Plot the layout
x_range = (-20, 20)
y_range = (-20, 20)
fig = plt.figure(figsize=(6.5*3, 6.5))

ax = fig.add_subplot(131, aspect='equal')
x = coords_3r0[:, 0]
y = coords_3r0[:, 1]
ax.plot(x, y, 'k.')
ax.set_xlim(x_range)
ax.set_ylim(y_range)
ax.set_title('3r0')
ax.grid()

ax = fig.add_subplot(132, aspect='equal')
x = coords_4r0[:, 0]
y = coords_4r0[:, 1]
ax.plot(x, y, 'k.')
ax.set_xlim(x_range)
ax.set_ylim(y_range)
ax.set_title('4r0')
ax.grid()

ax = fig.add_subplot(133, aspect='equal')
x = coords_4r0[:, 0]
y = coords_4r0[:, 1]
ax.plot(x, y, 'k.', markersize=10.0)
ax.set_xlim(x_range)
ax.set_ylim(y_range)
ax.set_title('4r0')
ax.grid()
coords_old = numpy.copy(coords_3r0)
coords_old[:, 1] = -coords_old[:, 1]
r0 = 48.0
for r in numpy.arange(r0 - 2.0, r0, 0.1):
    coords_new = rotate_station(math.radians(float(r)), coords_old)
    x = coords_new[:, 0]
    y = coords_new[:, 1]
    ax.plot(x, y, 'b.', alpha=0.5, markersize=4.0)
for r in numpy.arange(r0, r0 + 2.0, 0.1):
    coords_new = rotate_station(math.radians(float(r)), coords_old)
    x = coords_new[:, 0]
    y = coords_new[:, 1]
    ax.plot(x, y, 'g.', alpha=0.5, markersize=4.0)

coords_new = rotate_station(math.radians(float(r0)), coords_old)
x = coords_new[:, 0]
y = coords_new[:, 1]
ax.plot(x, y, 'r.')

plt.show()


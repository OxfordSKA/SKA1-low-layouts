import numpy
import matplotlib.pyplot as pyplot
from os.path import join
import math
import generate_v4a_super_station_layout as v4a_layout


def rotate_station(theta, coordinates):
    """ Rotate station coordinates.

    Args:
        theta (float): Rotation angle, in radians.
        coordinates (array_like): Station coordinates.

    Returns:
        rotated station coordinates.
    """
    rot = numpy.array([[math.cos(theta), math.sin(theta), 0.0],
                       [math.sin(theta), -math.cos(theta), 0.0],
                       [0.0, 0.0, 1.0]])
    return numpy.dot(coordinates, rot.T)


def rotate_layout(x, y, angle):
    theta = math.radians(angle)
    xrot = x * math.cos(theta) - y * math.sin(theta)
    yrot = y * math.sin(theta) + x * math.cos(theta)
    return xrot, yrot

# sbfile6r0 settings.
dalp = 0.  # Rotation angle of the entire station
alpg = [12., 20., 24., 34., 42., 57.]  # lattice rotation angles
angles = [-90., -54., 18., 90., 162., 234.]  # Sub-stations orientations.
sangles = [0., -18., 54., 126., 198., 270.]  # Station orientations.
fract_jitter = 0.0

sb6r0 = numpy.loadtxt(join('..', 'ant_files', 'sbfile6r0.ant'))
sb7r0 = numpy.loadtxt(join('..', 'ant_files', 'sbfile7r0.ant'))
x0, y0 = v4a_layout.generate(dalp, alpg, angles, sangles, fract_jitter)
x1, y1 = v4a_layout.generate(dalp, alpg, angles, sangles, fract_jitter)

# TODO-BM attempt to get sbfile7r0 by rotation of sbfile6r0
# vs direct generation of each ...


fig = pyplot.figure(figsize=(25, 12))

ax = fig.add_subplot(131, aspect='equal')
ax.plot(sb6r0[:, 0], sb6r0[:, 1], 'bx', label='sbfile6r0.ant')
ax.plot(x0.flatten(), y0.flatten(), 'r+', label='generated')
ax.set_xlim(-55, 55)
ax.set_ylim(-55, 55)
ax.legend()

ax = fig.add_subplot(132, aspect='equal')
ax.plot(sb7r0[:, 0], sb7r0[:, 1], 'bx', label='sbfile7r0.ant')
ax.plot(x1.flatten(), y1.flatten(), 'r+', label='generated')
ax.set_xlim(-55, 55)
ax.set_ylim(-55, 55)
ax.legend()


pyplot.show()



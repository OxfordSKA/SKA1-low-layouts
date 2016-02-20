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

sb6r0 = numpy.loadtxt(join('ant_files', 'sbfile6r0.ant'))
# sb7r0 = numpy.loadtxt(join('ant_files', 'sbfile7r0.ant'))
# x0, y0 = v4a_layout.generate(dalp, alpg, angles, sangles, fract_jitter)
# x1, y1 = v4a_layout.generate(dalp, alpg, angles, sangles, fract_jitter)
#
#
# fig = pyplot.figure(figsize=(20, 12))
#
# ax = fig.add_subplot(121, aspect='equal')
# ax.plot(sb6r0[:, 0], sb6r0[:, 1], 'bx', label2='sbfile6r0.ant')
# ax.plot(x0.flatten(), y0.flatten(), 'r+', label2='generated')
# ax.set_xlim(-55, 55)
# ax.set_ylim(-55, 55)
# ax.legend()
#
# ax = fig.add_subplot(122, aspect='equal')
# ax.plot(sb7r0[:, 0], sb7r0[:, 1], 'bx', label2='sbfile7r0.ant')
# ax.plot(x1.flatten(), y1.flatten(), 'r+', label2='generated')
# ax.set_xlim(-55, 55)
# ax.set_ylim(-55, 55)
# ax.legend()
#
#
# pyplot.show()
#

coords = numpy.loadtxt(join('layouts',
                            'v7ska1lowN1v2arev3R.enu.564x4.txt'))
coords = coords[:, 1:]

num_stations = coords.shape[0]
fig = pyplot.figure(figsize=(10.0, 10.0))
ax = fig.add_subplot(111, aspect='equal')

for i in range(num_stations/6):
    i0 = i*6
    i1 = i0 + 6
    sx0 = coords[i0, 0]
    sy0 = coords[i0, 1]
    xdiff = coords[i0 + 1, 0] - coords[i0 + 0, 0]
    ydiff = coords[i0 + 1, 1] - coords[i0 + 0, 1]
    ang = math.degrees(math.atan2(ydiff, xdiff))
    scoords = rotate_station(math.radians(ang + 90), sb6r0)
    ax.plot(scoords[:,0] + sx0, scoords[:,1] + sy0, '+',
            markersize=5.0)

for i in range(num_stations/6):
    i0 = i*6
    i1 = i0 + 6
    sx0 = coords[i0, 0]
    sy0 = coords[i0, 1]
    xdiff = coords[i0 + 1, 0] - coords[i0 + 0, 0]
    ydiff = coords[i0 + 1, 1] - coords[i0 + 0, 1]
    ang = math.degrees(math.atan2(ydiff, xdiff))
    ax.text(coords[i0, 0], coords[i0, 1] + 3.0,
            '%i: %.1f' % (i, ang),
            horizontalalignment='center', verticalalignment='bottom',
            fontsize='x-small')
    scoords = rotate_station(math.radians(ang + 90), sb6r0)
    ax.plot(coords[i0:i0 + 2, 0], coords[i0:i0 + 2, 1], 'o', color='r',
            markeredgecolor='k', markersize=5.0)
    ax.plot(coords[i0 + 2:i1, 0], coords[i0 + 2:i1, 1], 'o', color='k',
            markeredgecolor='k', markersize=5.0)

ax.set_xlim(-500, 500)
ax.set_ylim(-500, 500)
ax.set_xlabel('East [m]')
ax.set_ylabel('North [m]')
pyplot.show()

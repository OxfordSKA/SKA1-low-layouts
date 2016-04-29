"""Layout made by rotating sbfile6r0.ant to fit station positions."""

import numpy
import matplotlib.pyplot as pyplot
from os.path import join
from math import cos, sin, atan2, radians, degrees


def rotate_station(theta, coordinates):
    rot = numpy.array([[cos(theta), sin(theta), 0.0],
                       [sin(theta), -cos(theta), 0.0],
                       [0.0, 0.0, 1.0]])
    return numpy.dot(coordinates, rot.T)


sb6r0 = numpy.loadtxt(join('..', 'ant_files', 'sbfile6r0.ant'))
coords = numpy.loadtxt(join('..', 'layouts', 'v7ska1lowN1v2arev3R.enu.564x4.txt'))
coords = coords[:, 1:]

num_stations = coords.shape[0]
fig = pyplot.figure(figsize=(10.0, 10.0))
ax = fig.add_subplot(111, aspect='equal')

for i in range(num_stations/6):
    i0 = i*6
    i1 = i0 + 6
    sx0 = coords[i0, 0]
    sy0 = coords[i0, 1]
    x_diff = coords[i0 + 1, 0] - coords[i0 + 0, 0]
    y_diff = coords[i0 + 1, 1] - coords[i0 + 0, 1]
    ang = degrees(atan2(y_diff, x_diff))
    s_coords = rotate_station(radians(ang + 90), sb6r0)
    ax.plot(s_coords[:, 0] + sx0, s_coords[:, 1] + sy0, '+', markersize=5.0)

for i in range(num_stations/6):
    i0 = i*6
    i1 = i0 + 6
    sx0 = coords[i0, 0]
    sy0 = coords[i0, 1]
    x_diff = coords[i0 + 1, 0] - coords[i0 + 0, 0]
    y_diff = coords[i0 + 1, 1] - coords[i0 + 0, 1]
    ang = degrees(atan2(y_diff, x_diff))
    ax.text(coords[i0, 0], coords[i0, 1] + 3.0, '%i: %.1f' % (i, ang),
            horizontalalignment='center', verticalalignment='bottom',
            fontsize='x-small')
    s_coords = rotate_station(radians(ang + 90), sb6r0)
    ax.plot(coords[i0:i0 + 2, 0], coords[i0:i0 + 2, 1], 'o', color='r',
            markeredgecolor='k', markersize=5.0)
    ax.plot(coords[i0 + 2:i1, 0], coords[i0 + 2:i1, 1], 'o', color='k',
            markeredgecolor='k', markersize=5.0)

# ax.set_xlim(-500, 500)
# ax.set_ylim(-500, 500)
ax.set_xlabel('East [m]')
ax.set_ylabel('North [m]')
pyplot.show()

import numpy
import matplotlib
import matplotlib.pyplot as plt
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
dalp = 10.  # Rotation angle of the entire station
alpg = [12., 20., 24., 34., 42., 57.]  # lattice rotation angles
angles = [-90., -54., 18., 90., 162., 234.]  # Sub-stations orientations.
sangles = [0., -18., 54., 126., 198., 270.]  # Station orientations.
fract_jitter = 0.0

sb6r0 = numpy.loadtxt('sbfile6r0.ant')
ant_x0, ant_y0 = v4a_layout.generate(dalp, alpg, angles, sangles, fract_jitter)

# FIXME-BM: there is a bug somehwere in the v4a code...!!
fig = plt.figure(figsize=(10.0, 10.0))
ax = fig.add_subplot(111, aspect='equal')
ax.plot(ant_x0.flatten(), ant_y0.flatten(), 'r+')
scoords = rotate_station(math.radians(dalp + 90), sb6r0)
scoords = sb6r0
ax.plot(scoords[:, 0], scoords[:, 1], 'bx')
plt.show()

# # coords = numpy.loadtxt(join('SKA1_low_v4a_regular.tm', 'layout_enu.txt'))
# coords = numpy.loadtxt('v7ska1lowN1v2arev3R.enu.564x4.txt')
# coords = coords[:, 1:]
#
# num_stations = coords.shape[0]
# fig = plt.figure(figsize=(10.0, 10.0))
# ax = fig.add_subplot(111, aspect='equal')
# # ax.plot(coords[:, 0], coords[:, 1], 'k+')
#
# for i in range(num_stations/6):
#     i0 = i*6
#     i1 = i0 + 6
#     sx0 = coords[i0, 0]
#     sy0 = coords[i0, 1]
#     xdiff = coords[i0 + 1, 0] - coords[i0 + 0, 0]
#     ydiff = coords[i0 + 1, 1] - coords[i0 + 0, 1]
#     ang = math.degrees(math.atan2(ydiff, xdiff))
#     scoords = rotate_station(math.radians(ang + 90), sb6r0)
#     # ax.plot(scoords[:,0] + sx0, scoords[:,1] + sy0, '+', color='b',
#     #         markersize=4.0)
#     ax.plot(scoords[:,0] + sx0, scoords[:,1] + sy0, '+',
#             markersize=5.0)
#
# for i in range(num_stations/6):
#     i0 = i*6
#     i1 = i0 + 6
#     sx0 = coords[i0, 0]
#     sy0 = coords[i0, 1]
#     xdiff = coords[i0 + 1, 0] - coords[i0 + 0, 0]
#     ydiff = coords[i0 + 1, 1] - coords[i0 + 0, 1]
#     ang = math.degrees(math.atan2(ydiff, xdiff))
#     ax.text(coords[i0, 0], coords[i0, 1] + 3.0,
#             '%i: %.1f' % (i, ang),
#             horizontalalignment='center', verticalalignment='bottom',
#             fontsize='x-small')
#     scoords = rotate_station(math.radians(ang + 90), sb6r0)
#     ax.plot(coords[i0:i0 + 2, 0], coords[i0:i0 + 2, 1], 'o', color='r',
#             markeredgecolor='k', markersize=5.0)
#     ax.plot(coords[i0 + 2:i1, 0], coords[i0 + 2:i1, 1], 'o', color='k',
#             markeredgecolor='k', markersize=5.0)
#
# #ax.set_xlim(24934, 25077)
# # ax.set_ylim(14070, 14190)
# ax.set_xlim(-300, 300)
# ax.set_ylim(-300, 300)
# ax.set_xlabel('East [m]')
# ax.set_ylabel('North [m]')
# plt.savefig('layout_v4a_core_zoom.png')
# plt.show()

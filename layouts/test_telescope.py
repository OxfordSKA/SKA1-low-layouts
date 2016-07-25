# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from utilities.telescope import Telescope
from utilities.analysis import TelescopeAnalysis, SKA1_v5
import matplotlib.pyplot as plt


# tel = Telescope()
# tel.add_uniform_core(200, 1000)
# fig = tel.create_figure()
# ax = fig.gca()
# ax.add_artist(plt.Circle((0, 0), radius=1000, fill=False, color='r'))
# plt.show()


# tel = Telescope()
# tel.add_uniform_core(200, 1000)
# tel.add_ska1_v5(r_min=1000)
# tel.add_ska1_v5(r_min = 500, r_max = 1000)
# fig = tel.create_figure()
# ax = fig.gca()
# ax.add_artist(plt.Circle((0, 0), radius=500, fill=False, color='r'))
# ax.add_artist(plt.Circle((0, 0), radius=1000, fill=False, color='r'))
# plt.show()


# tel = Telescope()
# tel.add_uniform_core(368, 5000)
# tel.add_ska1_v5(r_min=5000)
# x, y, z = tel.coords()
# fig, ax = plt.subplots()
# for xy in zip(x, y):
#     ax.add_artist(plt.Circle(xy, radius=(tel.station_diameter_m / 2),
#                              fill=False))
# ax.set_aspect('equal')
# ax.set_xlim(-8000, 8000)
# ax.set_ylim(-8000, 8000)
# plt.show()
# # fig = tel.create_figure(plot_r=8000)
# # ax = fig.gca()
# # ax.add_artist(plt.Circle((0, 0), radius=1000, fill=False, color='r',
# #                          linestyle='--'))
# # plt.show()


# ska1 = SKA1_v5()
# ska1.create_figure(plot_r = 10000)
# plt.show()

tel = TelescopeAnalysis()
tel.add_ska1_v5(r_min=1000, r_max=5000)
# tel.add_uniform_core(368, 5000)

fig = tel.create_figure()
fig.savefig('zzz_layout.png')

tel.lon_deg = 116.631289
tel.lat_deg = -26.697024
tel.dec_deg = tel.lat_deg
tel.num_times = 10
tel.obs_length_h = 1
tel.gen_uvw_coords()

fig, ax = plt.subplots()
ax.plot(tel.uu_m, tel.vv_m, 'k.', ms=2.0, alpha=0.1)
ax.plot(-tel.uu_m, -tel.vv_m, 'k.', ms=2.0, alpha=0.1)
ax.set_aspect('equal')
ax.set_xlim(-10000, 10000)
ax.set_ylim(-10000, 10000)
fig.savefig('zzz_%ih.png' % tel.obs_length_h)
# plt.show()

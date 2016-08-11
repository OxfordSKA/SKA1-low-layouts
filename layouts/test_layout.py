# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from utilities.layout import Layout
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from math import cos, sin, radians


if __name__ == '__main__':
    layout = Layout(seed=3, num_trials=1, trail_timeout=3)

    # Random uniform coordinates
    # layout.uniform_cluster(300, 35, 500, 100)
    # layout.plot(plot_radii=[(100, 'r'), (500, 'b')])

    layout.uniform_cluster(5000, 1, 500, 0)
    # fig, ax = plt.subplots()
    # ax.plot(layout.x, layout.y, 'k+')
    n = 6
    theta0 = 360 // n
    theta0 = 90 + 360 // n
    layout.apply_poly_mask(n, 150, theta0, pad_radius=-20, invert=True)
    # layout.apply_poly_mask(n, 300, theta0, invert=False)
    # ax.plot(layout.x, layout.y, 'rx')
    # plt.show()
    layout.plot(plot_radius=500)

    # Hexagonal lattice
    layout.clear()
    layout.hex_lattice(35, 200, 0)
    layout.apply_poly_mask(6, 150, pad_radius=35/2, invert=False)
    layout.plot(plot_radius=200, plot_radii=[200])

    # hex = layout.hex_mask(40, offset=(30, 15))




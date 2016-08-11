# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from utilities.telescope import Telescope
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # # Uniform core with hole in the middle
    # tel = Telescope('test')
    # tel.add_uniform_core(300, 500, 100)
    # tel.plot(plot_radii=[100, 500])
    #
    # # Uniform core
    # tel = Telescope('test')
    # tel.add_uniform_core(250, 400)
    # tel.plot(plot_radii=[400])
    #
    # # Hexagonal core + Uniform halo
    tel = Telescope('test')
    tel.add_hex_core(200, 0.0)
    tel.add_uniform_core(180, 400, 200)
    tel.plot(plot_radii=[200, 400])

    # Spiral arms
    tel = Telescope('test')
    tel.add_log_spiral(n=3*10, r0=100, r1=400, b=0.5, num_arms=3, theta0_deg=0)
    tel.plot(plot_radii=[100, 400])

    # Symmetric spiral arms
    tel = Telescope('test')
    tel.add_symmetric_log_spiral(n=10, r0=100, r1=400, b=0.5, num_arms=3,
                                 name='foo', theta0_deg=0)
    tel.plot(plot_radii=[100, (400, 'b')], color='g')


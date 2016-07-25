# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
from . import telescope
from pyuvwsim import (evaluate_baseline_uvw_ha_dec, convert_enu_to_ecef)
from math import pi, radians, degrees


class TelescopeAnalysis(telescope.Telescope):
    def __init__(self):
        telescope.Telescope.__init__(self)
        self.obs_length_h = 0
        self.dec_deg = 0
        self.num_times = 1
        self.uu_m = None
        self.vv_m = None
        self.ww_m = None

    def gen_uvw_coords(self):
        """Generate uvw coordinates"""
        x, y, z = self.coords_enu()
        x, y, z = convert_enu_to_ecef(x, y, z, radians(self.lon_deg),
                                      radians(self.lat_deg), self.alt_m)
        num_stations = x.shape[0]
        num_baselines = num_stations * (num_stations - 1) // 2
        n = num_baselines * self.num_times
        self.uu_m, self.vv_m, self.ww_m = np.zeros(n), np.zeros(n), np.zeros(n)
        ha_off = ((self.obs_length_h / 2) / 24) * (2 * pi)
        for i, ha in enumerate(np.linspace(-ha_off, ha_off, self.num_times)):
            uu_, vv_, ww_ = evaluate_baseline_uvw_ha_dec(
                x, y, z, ha - radians(self.lon_deg), radians(self.dec_deg))
            self.uu_m[i * num_baselines: (i + 1) * num_baselines] = uu_
            self.vv_m[i * num_baselines: (i + 1) * num_baselines] = vv_
            self.ww_m[i * num_baselines: (i + 1) * num_baselines] = ww_

    def uvw_grid(self):
        pass

    def uv_hist(self):
        pass

    def network_graph(self):
        pass

    def psfrms(self):
        pass

    def uvgap(self):
        pass

    def psf(self):
        pass


class SKA1_v5(TelescopeAnalysis):
    def __init__(self):
        TelescopeAnalysis.__init__(self)
        self.add_ska1_v5()
        self.lon_deg = 116.63128900
        self.lat_deg = -26.69702400

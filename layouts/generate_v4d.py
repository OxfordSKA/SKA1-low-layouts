# -*- coding: utf-8 -*-
"""Script to generate v4d station and super-station coordinates.

Based on v4d description in the document:
    'Objectives for a Proposed SKA1-Low Station-Configuration Workshop'
    P. Dewdney, J. Wagg, R. Braun, M.-G. Labate
    Feb 1st 2016
which was emailed prior to the SKA1-Low Station-Configuration Workshop
held on the 25-26th Feb 2016

Changes:
    29/02/2016: Initial version.
"""
from __future__ import print_function
import numpy
import matplotlib.pyplot as pyplot
try:
    from pyuvwsim import (load_station_coords, convert_enu_to_ecef,
                          evaluate_baseline_uvw)
    uvwsim_found = True
except ImportError:
    print('pyuvwsim not found, skipping uvw co-ordiante generation.')
    print('see: https://github.com/SKA-ScienceDataProcessor/uvwsim, pyuvwsim.rst')
    uvwsim_found = False
from math import radians


def rotate_coords(x, y, angle):
    """Rotate array of x, y coordinates counter clockwise by angle, in deg."""
    xr = x * numpy.cos(radians(angle)) - y * numpy.sin(radians(angle))
    yr = x * numpy.sin(radians(angle)) + y * numpy.cos(radians(angle))
    return xr, yr


def generate_baseline_uvw(x, y, z, ra_rad, dec_rad, num_times, num_baselines,
                          mjd_start, dt_s):
    """Generate baseline coordinates from ecef station coordinates."""
    num_coords = num_times * num_baselines
    uu = numpy.zeros(num_coords, dtype='f8')
    vv = numpy.zeros(num_coords, dtype='f8')
    ww = numpy.zeros(num_coords, dtype='f8')
    for i in range(num_times):
        t = i * dt_s + dt_s / 2.0
        mjd = mjd_start + (t / 86400.0)
        i0 = i * num_baselines
        i1 = i0 + num_baselines
        uu_, vv_, ww_ = evaluate_baseline_uvw(x, y, z, ra_rad, dec_rad, mjd)
        uu[i0:i1] = uu_
        vv[i0:i1] = vv_
        ww[i0:i1] = ww_
    return uu, vv, ww


def main():
    # ==========================================================================
    ring_counts = [1, 5, 11, 17]
    ring_radii = [0.0, 100.0, 190.0, 290.0]  # metres
    ring_start_angle = numpy.random.randint(0, 360, 4)
    a = 300.0
    b = 0.513
    delta_theta = 37.0
    num_arms = 3
    arm_count = 5
    arm_offsets = [35.0, 155.0, 275.0]
    ss_radius = 90.0 / 2.0
    st_radius = 35.0 / 2.0
    num_stations_per_ss = 6
    num_super_stations = numpy.sum(numpy.array(ring_counts)) + \
                         num_arms * arm_count
    v4d_ss_x = numpy.zeros(num_super_stations)
    v4d_ss_y = numpy.zeros(num_super_stations)
    ss_petal_angle = -360.0 * numpy.random.random(num_super_stations) + 360.0
    # ==========================================================================

    # Generate super-station rings
    idx = 0
    for i, n in enumerate(ring_counts):
        angles = numpy.arange(n) * (360.0 / n)
        angles += ring_start_angle[i]
        x = ring_radii[i] * numpy.cos(numpy.radians(angles))
        y = ring_radii[i] * numpy.sin(numpy.radians(angles))
        v4d_ss_x[idx:idx+n] = x
        v4d_ss_y[idx:idx+n] = y
        idx += n

    # Generate super-station spiral arms
    for i in range(num_arms):
        t = numpy.arange(1, arm_count + 1) * delta_theta
        t = numpy.radians(t)
        x = a * numpy.exp(b * t) * numpy.cos(t + numpy.radians(arm_offsets[i]))
        y = a * numpy.exp(b * t) * numpy.sin(t + numpy.radians(arm_offsets[i]))
        v4d_ss_x[idx:idx+arm_count] = x
        v4d_ss_y[idx:idx+arm_count] = y
        idx += arm_count

    v4d_st_x = numpy.zeros((num_super_stations, num_stations_per_ss))
    v4d_st_y = numpy.zeros((num_super_stations, num_stations_per_ss))

    # Generate stations
    for i in range(num_super_stations):
        angles = 360.0 / (num_stations_per_ss - 1) * \
                 numpy.arange(num_stations_per_ss - 1)
        x = (st_radius * 2.0) * numpy.cos(numpy.radians(angles))
        y = (st_radius * 2.0) * numpy.sin(numpy.radians(angles))
        x, y = rotate_coords(x, y, ss_petal_angle[i])
        v4d_st_x[i, 1:] = x
        v4d_st_y[i, 1:] = y
        v4d_st_x[i, :] += v4d_ss_x[i]
        v4d_st_y[i, :] += v4d_ss_y[i]

    v4d_st_x = v4d_st_x.flatten()
    v4d_st_y = v4d_st_y.flatten()

    num_stations = num_super_stations * num_stations_per_ss
    v4d_st_enu = numpy.zeros((num_super_stations * num_stations_per_ss, 3))
    v4d_st_enu[:, 0] = v4d_st_x
    v4d_st_enu[:, 1] = v4d_st_y
    numpy.savetxt('v4d_stations_enu.txt', v4d_st_enu,
                  fmt='% -16.12f % -16.12f % -16.12f')

    v4d_ss_enu = numpy.zeros((num_super_stations, 3))
    v4d_ss_enu[:, 0] = v4d_ss_x
    v4d_ss_enu[:, 1] = v4d_ss_y
    numpy.savetxt('v4d_super_stations_enu.txt', v4d_ss_enu,
                  fmt='% -16.12f % -16.12f % -16.12f')

    # ==== Plotting ===========================================================
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(v4d_ss_x, v4d_ss_y, 'k+')
    for i in range(num_super_stations - 12):
        circle = pyplot.Circle((v4d_ss_x[i], v4d_ss_y[i]),
                               ss_radius, color='b',
                               fill=True, alpha=0.5)
        ax.add_artist(circle)

    for i in range(num_super_stations - 15, num_super_stations - 10):
        circle = pyplot.Circle((v4d_ss_x[i], v4d_ss_y[i]),
                               ss_radius, color='r',
                               fill=True, alpha=0.5)
        ax.add_artist(circle)

    for i in range(num_super_stations - 10, num_super_stations - 5):
        circle = pyplot.Circle((v4d_ss_x[i], v4d_ss_y[i]),
                               ss_radius, color='g',
                               fill=True, alpha=0.5)
        ax.add_artist(circle)

    for i in range(num_super_stations - 5, num_super_stations):
        circle = pyplot.Circle((v4d_ss_x[i], v4d_ss_y[i]),
                               ss_radius, color='y',
                               fill=True, alpha=0.5)
        ax.add_artist(circle)

    for i in range(num_super_stations * num_stations_per_ss):
        circle = pyplot.Circle((v4d_st_x[i], v4d_st_y[i]),
                               st_radius, color='k',
                               fill=True, alpha=0.5)
        ax.add_artist(circle)

    circle = pyplot.Circle((0.0, 0.0), 1700.0, color='k', linestyle=':',
                           linewidth=2.0,
                           fill=False, alpha=0.5)
    ax.add_artist(circle)

    ax.set_xlim(-2000, 2000)
    ax.set_ylim(-2000, 2000)

    # and a corresponding grid
    ax.grid(which='both')

    # or if you want differnet settings for the grids:
    ax.grid(which='minor', alpha=0.5)
    ax.grid(which='major', alpha=1.0)
    ax.set_ylabel('North [m]')
    ax.set_xlabel('East [m]')
    pyplot.savefig('v4d_station_layout.png')
    pyplot.show()

    if uvwsim_found:
        x = v4d_st_x
        y = v4d_st_y
        z = numpy.zeros_like(v4d_st_x)
        lon = 116.63128900
        lat = -26.69702400
        alt = 0.0
        # Zeith
        ra = 68.698903779331502
        dec = -26.568851215532160
        # 67.5 elevation
        # ra = 68.662690336853558
        # dec = -4.070637553699950
        # 45.0 elevation
        # ra = 68.628471594503878
        # dec = 18.427199372436142
        mjd_start = 57443.4375000000
        dt_s = 0.0
        num_times = 1
        num_baselines = num_stations * (num_stations - 1) / 2
        freq_hz = 150e6
        x, y, z = convert_enu_to_ecef(x, y, z, lon, lat, alt)
        uu, vv, ww = generate_baseline_uvw(x, y, z, ra, dec, num_times,
                                           num_baselines, freq_hz, mjd_start,
                                           dt_s)

        fig = pyplot.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, aspect='equal')
        ax.plot(uu, vv, 'k.', alpha=0.5, ms=2.0)
        ax.plot(-uu, -vv, 'k.', alpha=0.5, ms=2.0)
        ax.set_xlim(-3000, 3000)
        ax.set_ylim(-3000, 3000)
        ax.set_xlabel('uu [m]')
        ax.set_ylabel('vv [m]')
        pyplot.savefig('v4d_snapshot_uv_zenith.png')
        pyplot.show()

        fig = pyplot.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, aspect='equal')
        ax.plot(uu, vv, 'k.', alpha=0.5, ms=2.0)
        ax.plot(-uu, -vv, 'k.', alpha=0.5, ms=2.0)
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(-1000, 1000)
        ax.set_xlabel('uu [m]')
        ax.set_ylabel('vv [m]')
        pyplot.savefig('v4d_snapshot_uv_zenith_zoom.png')
        pyplot.show()


if __name__ == '__main__':
    main()

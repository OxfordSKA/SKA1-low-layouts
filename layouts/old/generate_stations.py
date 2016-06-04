# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import shutil
from os.path import join

import matplotlib.pyplot as pyplot
import numpy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.random import rand


def gridgen(num_points, diameter, min_dist, max_trials=1000):
    def grid_position(x, y, scale, grid_size):
        jx = int(round(x * scale)) + grid_size / 2
        jy = int(round(y * scale)) + grid_size / 2
        return jx, jy

    def get_trail_position(r):
        x = -r + 2.0 * r * rand()
        y = -r + 2.0 * r * rand()
        return x, y

    # Grid size and scaling onto the grid
    grid_size = min(100, int(round(float(diameter) / min_dist)))
    grid_cell = float(diameter) / grid_size  # Grid sector cell size
    scale = 1.0 / grid_cell  # Scaling onto the sector grid.
    check_width = 1

    r = diameter / 2.0  # Radius
    r_sq = r**2  # Radius, squared
    min_dist_sq = min_dist**2  # minimum distance, squared
    r_ant = min_dist / 2.0

    # Pre-allocate coordinate arrays
    x = numpy.zeros(num_points)
    y = numpy.zeros(num_points)

    # Grid meta-data
    grid_i_start = numpy.zeros((grid_size, grid_size), dtype='i8')
    grid_i_end = numpy.zeros((grid_size, grid_size), dtype='i8')
    grid_count = numpy.zeros((grid_size, grid_size), dtype='i8')
    grid_i_next = numpy.zeros(num_points, dtype='i8')

    n = num_points
    n_req = num_points
    num_tries = 0
    try_count = list()
    for j in range(n_req):

        done = False
        while not done:

            # Generate a trail position
            xt, yt = get_trail_position(r)
            rt = (xt**2 + yt**2)**0.5

            # Check if the point is inside the diameter.
            if rt + r_ant > r:
                num_tries += 1

            # Check if min distance is met.
            else:
                jx, jy = grid_position(xt, yt, scale, grid_size)
                y0 = max(0, jy - check_width)
                y1 = min(grid_size, jy + check_width + 1)
                x0 = max(0, jx - check_width)
                x1 = min(grid_size, jx + check_width + 1)
                d_min = diameter  # Set initial min to diameter.
                for ky in range(y0, y1):
                    for kx in range(x0, x1):
                        if grid_count[kx, ky] > 0:
                            kh1 = grid_i_start[kx, ky]
                            for kh in range(grid_count[kx, ky]):
                                dx = xt - x[kh1]
                                dy = yt - y[kh1]
                                d_min = min((dx**2 + dy**2)**0.5, d_min)
                                kh1 = grid_i_next[kh1]

                if d_min >= min_dist:
                    x[j] = xt
                    y[j] = yt
                    if grid_count[jx, jy] == 0:
                        grid_i_start[jx, jy] = j
                    else:
                        grid_i_next[grid_i_end[jx, jy]] = j
                    grid_i_end[jx, jy] = j
                    grid_count[jx, jy] += 1
                    try_count.append(num_tries)
                    num_tries = 0
                    done = True
                else:
                    num_tries += 1

            if num_tries >= max_trials:
                n = j - 1
                done = True

        if num_tries >= max_trials:
            break

    if n < n_req:
        x = x[0:n]
        y = y[0:n]

    return x, y, try_count


def taylor_win(n, nbar=4, sll=-30):
    """
    http://www.dsprelated.com/showcode/6.php

    from http://mathforum.org/kb/message.jspa?messageID=925929:

    A Taylor window is very similar to Chebychev weights. While Chebychev
    weights provide the tighest beamwidth for a given side-lobe level, taylor
    weights provide the least taper loss for a given sidelobe level.

    'Antenna Theory: Analysis and Design' by Constantine Balanis, 2nd
    edition, 1997, pp. 358-368, or 'Modern Antenna Design' by Thomas
    Milligan, 1985, pp.141-148.
    """
    def calculate_fm(m, sp2, a, nbar):
        n = numpy.arange(1, nbar)
        p = numpy.hstack([numpy.arange(1, m, dtype='f8'),
                          numpy.arange(m+1, nbar, dtype='f8')])
        num = numpy.prod((1 - (m**2/sp2) / (a**2 + (n - 0.5)**2)))
        den = numpy.prod(1.0 - m**2 / p**2)
        fm = ((-1)**(m + 1) * num) / (2.0 * den)
        return fm
    a = numpy.arccosh(10.0**(-sll/20.0))/numpy.pi
    sp2 = nbar**2 / (a**2 + (nbar - 0.5)**2)
    w = numpy.ones(n)
    fm = numpy.zeros(nbar)
    summation = 0
    k = numpy.arange(n)
    xi = (k - 0.5 * n + 0.5) / n
    for m in range(1, nbar):
        fm[m] = calculate_fm(m, sp2, a, nbar)
        summation = fm[m] * numpy.cos(2.0 * numpy.pi * m * xi) + summation
    w += 2.0 * summation
    return w


def taylor(x, y, sll=-28):
    nbar = int(numpy.ceil(2.0 * (numpy.arccosh(10**(-sll / 20.0)) /
                                 numpy.pi)**2+0.5))
    w_taylor = taylor_win(10000, nbar, sll)
    w_taylor = w_taylor[5000:]
    x0 = numpy.copy(x)
    y0 = numpy.copy(y)
    x0 -= (x0.min() + x0.max()) / 2.0
    y0 -= (y0.min() + y0.max()) / 2.0
    r_vec = numpy.sqrt(x0**2 + y0**2)
    rr = numpy.linspace(0.0001, r_vec.max()+0.001, 5000)

    ab = numpy.ones(x.shape[0], dtype='i8')
    for ii in range(x.shape[0]):
        r_dif = numpy.abs(r_vec[ii] - rr)
        bb = numpy.where(r_dif == r_dif.min())
        ab[ii] = bb[0]

    w_ch = w_taylor[ab]
    w_ch = w_ch / w_ch.max()

    return w_ch


def generate_apodisation_file(station_file, sll):
    layout = numpy.loadtxt(station_file)
    x = layout[:, 0]
    y = layout[:, 1]
    w = taylor(x, y, sll)
    print(numpy.sum(w), x.shape[0])
    pyplot.scatter(x, y, s=20, c=w)
    pyplot.show()
    numpy.savetxt('TEMP.txt', w)


def main():
    tel_dir = 'stations.tm'
    if os.path.isdir(tel_dir):
        shutil.rmtree(tel_dir)
    max_tries_per_station = 5
    num_stations = 512
    num_antennas = 256
    station_diameter = 35.0  # m
    antenna_diameter = 1.5  # m
    sll = -28.0
    station_coords = numpy.zeros((num_antennas, 2), dtype='f8')

    fig1 = pyplot.figure(figsize=(8, 8))
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('North [m]')
    ax1.grid()
    ax1.set_xlim(-20.0, 20.0)
    ax1.set_ylim(-20.0, 20.0)
    line1, = ax1.plot([], [], 'k+')
    circle1 = pyplot.Circle((0.0, 0.0), station_diameter / 2.0, color='r',
                            linestyle='--', fill=False, alpha=0.3, lw=1.0)
    ax1.add_artist(circle1)

    fig2 = pyplot.figure(figsize=(8, 8))
    ax2 = fig2.add_subplot(111, aspect='equal')
    ax2.set_xlabel('East [m]')
    ax2.set_ylabel('North [m]')
    ax2.grid()
    ax2.set_xlim(-20.0, 20.0)
    ax2.set_ylim(-20.0, 20.0)
    circle2 = pyplot.Circle((0.0, 0.0), station_diameter / 2.0, color='r',
                            linestyle='--', fill=False, alpha=0.3, lw=1.0)
    ax2.add_artist(circle2)

    for i in range(num_stations):
        trial = 0
        print('Generating station %i ' % i, end='')
        while trial < max_tries_per_station:
            n_gen = 384
            d_gen = 42
            ax, ay, _ = gridgen(n_gen, d_gen, antenna_diameter,
                                max_trials=10000)
            if ax.shape[0] == n_gen:
                ar = (ax**2 + ay**2)**0.5
                sort_idx = ar.argsort()
                ax = ax[sort_idx]
                ay = ay[sort_idx]
                ar = ar[sort_idx]
                ax = ax[:num_antennas]
                ay = ay[:num_antennas]
                ar = ar[:num_antennas]
                ax = numpy.append(ax, 0.0)
                ay = numpy.append(ay, 0.0)
                w = taylor(ax, ay, sll)
                ax = ax[:num_antennas]
                ay = ay[:num_antennas]
                w = w[:num_antennas]

                station_dir = join(tel_dir, 'station%03i' % i)
                station_file = join(station_dir, 'layout.txt')
                apod_file = join(station_dir, 'apodisation.txt')
                rad_file = join(station_dir, 'radius.txt')

                if not os.path.isdir(station_dir):
                    os.makedirs(station_dir)

                numpy.savetxt(apod_file, w, fmt='% -16.12f')
                station_coords[:, 0] = ax
                station_coords[:, 1] = ay
                numpy.savetxt(station_file, station_coords,
                              fmt='% -16.12f % -16.12f')
                numpy.savetxt(rad_file, ar, fmt='% -16.12f')

                line1.set_data(ax, ay)
                if i == num_stations - 1:
                    for i_ant in range(num_antennas):
                        circle = pyplot.Circle((ax[i_ant], ay[i_ant]),
                                               antenna_diameter / 2.0,
                                               color='b', fill=True, alpha=0.3,
                                               lw=0.0)
                        ax1.add_artist(circle)
                fig1.savefig(join(station_dir, 'station%03i.png' % i))
                ax2.plot(ax, ay, 'k+', alpha=0.1)
                if i == 0:
                    fig3 = pyplot.figure(figsize=(8, 8))
                    ax3 = fig3.add_subplot(111, aspect='equal')
                    divider = make_axes_locatable(ax3)
                    cax3 = divider.append_axes("right", size="3%", pad=0.07)
                    ax3.set_xlabel('East [m]')
                    ax3.set_ylabel('North [m]')
                    ax3.grid()
                    ax3.set_xlim(-20.0, 20.0)
                    ax3.set_ylim(-20.0, 20.0)
                    circle3 = pyplot.Circle((0.0, 0.0), station_diameter / 2.0,
                                            color='k',
                                            linestyle='--', fill=False,
                                            alpha=0.3, lw=1.0)
                    ax3.add_artist(circle3)
                    sc = ax3.scatter(ax, ay, s=70, c=w, lw=0.0, cmap='inferno')
                    cbar3 = ax3.figure.colorbar(sc, cax=cax3)
                    cbar3.set_label('Apodisation weight')
                    fig3.savefig(join(station_dir, 'apodisation.png'))

                print('Done(%i)' % trial)
                break
            else:
                trial += 1
                print('.', end='')
                continue
        if trial == max_tries_per_station:
            print()
            print('Error, failed to find enough antennas for station '
                  '%i/%i' % (ax.shape[0], num_antennas))
            return
        fig2.savefig(join(tel_dir, 'all_stations.png'))
        pyplot.close(fig1)
        pyplot.close(fig2)

if __name__ == '__main__':
    main()

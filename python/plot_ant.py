import numpy
import matplotlib.pyplot as plt
import os

ant_files = [f for f in os.listdir('.') if f.endswith('.ant')]
for ant_file in ant_files:
    layout = numpy.loadtxt(ant_file)
    ok = layout[:, 0] != -1.0e6
    layout = layout[ok, :]
    if ant_file == 'sbfile3r0.ant' or ant_file == 'sbfile4r0.ant':
        fig = plt.figure(figsize=(6.5, 6.5))
        ax = fig.add_subplot(111, aspect='equal')
        total_antennas = layout.shape[0]
        sub_station_size = total_antennas / 6
        for s in range(6):
            i0 = s * sub_station_size
            i1 = i0 + sub_station_size
            x = layout[i0:i1, 0]
            y = layout[i0:i1, 1]
            ax.plot(x, y, '.')
            ax.grid(True)
            ax.set_ylim(-25, 25)
            ax.set_xlim(-25, 25)
        ax.set_title('%s (%i dipoles)' % (ant_file, layout.shape[0]))
        ax.set_xlabel('east [m]')
        ax.set_ylabel('north [m]')
        plt.savefig('%s.png' % ant_file)
    else:
        fig = plt.figure(figsize=(6.5, 6.5))
        ax = fig.add_subplot(111, aspect='equal')
        x = layout[:, 0]
        y = layout[:, 1]
        ax.plot(x, y, '+')
        ax.set_xlabel('east [m]')
        ax.set_ylabel('north [m]')
        ax.set_title('%s (%i dipoles)' % (ant_file, layout.shape[0]))
        # plt.show()
        plt.savefig('%s.png' % ant_file)

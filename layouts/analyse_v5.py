from utilities.telescope import Telescope
from utilities.analysis import SKA1_low_analysis
from math import degrees, atan2, radians, cos, sin, exp, fmod, pi
import matplotlib.pyplot as plt
import numpy as np
from os import makedirs
from os.path import isdir, join


def main():
    # -------------- Options
    out_dir = 'results_01'
    # --------------

    if not isdir(out_dir):
        makedirs(out_dir)

    # Current SKA1 V5 design.
    tel = SKA1_low_analysis()
    tel.add_ska1_v5(None, 6400)
    tel.plot_layout(plot_radii=[500, 6400], color='k',
                    filename=join(out_dir, 'layout_v5.png'))

    # Generate new telescopes by expanding each station cluster.
    b = 0.515
    theta0_deg = -48
    start_inner = 417.82
    num_arms = 3
    d_theta = 360 / num_arms

    # Get cluster radii.
    cluster_x, cluster_y, arm_index = Telescope.cluster_centres_ska_v5(0, 6400)
    cluster_r = (cluster_x**2 + cluster_y**2)**0.5
    cluster_r = cluster_r[::3]  # Get every 3rd radius.
    delta_theta_deg = Telescope.delta_theta(
        cluster_x[0], cluster_y[0], cluster_x[3], cluster_y[3], start_inner, b)

    # Loop over cluster radii.
    for i in range(len(cluster_r)):
        if i != 3:
            continue
        print('-' * 80)
        # Create the telescope and add the core.
        tel1 = SKA1_low_analysis()
        tel1.add_ska1_v5(None, 500)

        # Add SKA1 V5 clusters from this radius outwards.
        if i < len(cluster_r) - 1:
            r = cluster_r[i + 1]
            tel1.add_ska1_v5(r - 90, 6400)

        # Add spiral sections up to this radius.
        for j in range(i + 1):
            for k in range(num_arms):
                idx = num_arms * j + k
                tel1.add_log_spiral_section(
                    6, start_inner,
                    cluster_x[idx], cluster_y[idx],
                    b, delta_theta_deg / 2.0,
                    theta0_deg + arm_index[idx] * d_theta)
        # tel1.plot_layout(plot_radii=[500, 6400], color='k',
        #                  show_decorations=True,
        #                  filename=join(out_dir, 'layout_%02i.png' % i))
        # tel1.plot_grid(filename=join(out_dir, 'uv_grid_%02i.png' % i))
        # tel1.plot_network()
        tel1.cable_cost()

if __name__ == '__main__':
    main()

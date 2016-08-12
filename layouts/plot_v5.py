from utilities.telescope import Telescope
from math import degrees, atan2, radians, cos, sin, exp, fmod, pi
import matplotlib.pyplot as plt
import numpy as np


def test1():
    tel = Telescope()

    # Spiral parameters for inner and outer regions.
    start_inner = 417.82
    end_inner = 1572.13
    b_inner = 0.513
    start_outer = 2146.78
    end_outer = 6370.13
    b_outer = 0.52

    # Add all stations out to 6500 m.
    tel.add_ska1_v5(r_min=500, r_max=6500)

    # Add the cluster centres.
    tel.add_symmetric_log_spiral(5, start_inner, end_inner, b_inner,
                                 3, 'cluster_centres_inner', -48)
    tel.add_symmetric_log_spiral(5, start_outer, end_outer, b_outer,
                                 3, 'cluster_centres_outer', 135)

    # Fill in the gaps with spirals.
    b = 0.515
    tel.add_symmetric_log_spiral(60, start_inner, end_outer, b,
                                 3, 'spiral_arms', -48)

    tel.plot_layout(plot_radii=[start_inner, end_inner, start_outer, end_outer],
                    show_decorations=False)


def test2():
    # Spiral parameters for inner and outer regions.
    start_inner = 417.82
    end_outer = 6370.13

    b = 0.515
    theta0 = -40
    num_arms = 1
    tel1 = Telescope()
    tel1.add_symmetric_log_spiral(10, start_inner, end_outer, b,
                                  num_arms, 'spiral_arms', theta0)
    cx1 = tel1.layouts['spiral_arms0']['x'][3]
    cy1 = tel1.layouts['spiral_arms0']['y'][3]
    cx2 = tel1.layouts['spiral_arms0']['x'][4]
    cy2 = tel1.layouts['spiral_arms0']['y'][4]

    delta_t = Telescope.delta_theta(cx1, cy1, cx2, cy2, start_inner, b)

    tel2 = Telescope()
    tel2.add_log_spiral_section(6, start_inner, cx1, cy1, b, delta_t / 2,
                                num_arms, theta0)

    tel3 = Telescope()
    tel3.add_log_spiral_section(6, start_inner, cx2, cy2, b, delta_t / 2,
                                num_arms, theta0)

    fig, ax = plt.subplots(figsize=(8, 8))
    tel2.plot_layout(mpl_ax=ax, color='g')
    tel3.plot_layout(mpl_ax=ax, color='y')
    tel1.station_diameter_m = 50
    tel1.plot_layout(plot_radii=[start_inner, end_outer],
                     mpl_ax=ax, color='r')
    ax.plot([cx1], [cy1], '+', ms=10)
    plt.show()


def test3():
    # Current SKA1 V5 design.
    tel = Telescope()
    tel.add_ska1_v5(None, 6400)
    tel.plot_layout(plot_radii=[500, 6400], color='k')

    # Generate new telescopes by expanding each station cluster.
    b = 0.515
    theta0_deg = -48
    start_inner = 417.82
    num_arms = 3
    d_theta = 360 / num_arms

    # Get cluster radii.
    cluster_x, cluster_y, arm_index = Telescope.cluster_centres_ska_v5(0, 6400)
    cluster_r = (cluster_x**2 + cluster_y**2)**0.5
    cluster_r = cluster_r[::3] # Get every 3rd radius.
    delta_theta_deg = Telescope.delta_theta(
        cluster_x[0], cluster_y[0], cluster_x[3], cluster_y[3], start_inner, b)

    # Loop over cluster radii.
    for i in range(len(cluster_r)):
        # Create the telescope and add the core.
        tel1 = Telescope()
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
                    b, delta_theta_deg / 2.0, 1,
                    theta0_deg + arm_index[idx] * d_theta)
        # NOTE(BM) Telescope exists here ... add metrics
        tel1.plot_layout(plot_radii=[500, 6400], color='k', show_decorations=True)



test3()

from utilities.telescope import Telescope
from math import degrees, atan2, radians, cos, sin, exp, fmod, pi
import matplotlib.pyplot as plt
import numpy as np

def test1():
    tel = Telescope()

    # Spiral parameters for inner and outer regions.
    start_inner = 417.82
    end_inner = 1572.13
    spiral_parameter_inner = 0.513
    start_outer = 2146.78
    end_outer = 6370.13
    spiral_parameter_outer = 0.52

    # Add all stations out to 6500 m.
    tel.add_ska1_v5(r_min=500, r_max=6500)

    # Add the cluster centres.
    tel.add_symmetric_log_spiral(5, start_inner, end_inner, spiral_parameter_inner,
                                 3, 'cluster_centres_inner', -48)
    tel.add_symmetric_log_spiral(5, start_outer, end_outer, spiral_parameter_outer,
                                 3, 'cluster_centres_outer', 135)

    # Fill in the gaps with spirals.
    # tel.add_log_spiral(200, start_inner, end_inner,
    #                    spiral_parameter_inner, 3, -48)
    # tel.add_log_spiral(200, start_outer, end_outer,
    #                    spiral_parameter_outer, 3, 135)
    tel.add_symmetric_log_spiral(60, start_inner, end_outer + 0, 0.515,
                                 3, 'spiral_arms', -48)

    tel.plot_layout(plot_radii=[start_inner, end_inner, start_outer, end_outer],
                    plot_decorations=False)


def test2():
    tel1 = Telescope()

    # Spiral parameters for inner and outer regions.
    start_inner = 417.82
    end_outer = 6370.13

    theta0 = -40
    tel1.add_symmetric_log_spiral(10, start_inner, end_outer + 0, 0.515,
                                  1, 'spiral_arms', theta0)
    cx1 = tel1.layouts['spiral_arms0']['x'][3]
    cy1 = tel1.layouts['spiral_arms0']['y'][3]
    cr1 = (cx1**2 + cy1**2)**0.5

    cx2 = tel1.layouts['spiral_arms0']['x'][4]
    cy2 = tel1.layouts['spiral_arms0']['y'][4]

    delta_t = Telescope.delta_theta(cx1, cy1, cx2, cy2, start_inner, 0.515)
    print(delta_t)

    tel2 = Telescope()
    # tel2.add_log_sprial_section(12, start_inner, 1000, 1500, 0.515, 1, theta0)
    # tel2.add_log_sprial_section(100, start_inner, 5000, 6000, 0.515, 1, theta0)
    tel2.add_symmetric_log_spiral(60, start_inner, end_outer + 0, 0.515,
                                  1, 'spiral_arms', 0)

    tel3 = Telescope()
    tel3.add_log_sprial_section_2(6, start_inner, cx1, cy1, 0.515, delta_t/2,
                                  1, theta0)

    tel4 = Telescope()
    tel4.add_log_sprial_section_2(6, start_inner, cx2, cy2, 0.515, delta_t/2,
                                  1, theta0)
    # tel4.station_diameter_m = 17

    fig, ax = plt.subplots(figsize=(8, 8))
    tel3.plot_layout(mpl_ax=ax, station_color='g')
    tel4.plot_layout(mpl_ax=ax, station_color='y')
    # tel2.plot_layout(mpl_ax=ax, station_color='b')
    tel1.station_diameter_m = 50
    tel1.plot_layout(plot_radii=[start_inner, end_outer, 1000, 1500, cr1],
                     mpl_ax=ax, station_color='r')
    ax.plot(cx1, cy1, 'k+', ms=10, mew=1.5)
    ax.plot(cx2, cy2, 'k+', ms=10, mew=1.5)
    plt.show()


test2()

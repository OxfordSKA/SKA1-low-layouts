from utilities.telescope import Telescope

tel = Telescope()

# Spiral parameters for inner and outer regions.
start_inner = 417.82
end_inner = 1572.13
spiral_parameter_inner = 0.513
start_outer = 2146.78
end_outer = 6370.13
spiral_parameter_outer = 0.52

# Add all stations out to 6500 m.
tel.add_ska1_v5(r_min=500,r_max=6500)

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
tel.add_symmetric_log_spiral(60, start_inner, end_outer + 1000, 0.515,
                             3, 'spiral_arms', -48)


tel.plot_layout(plot_radii=[start_inner, end_inner, start_outer, end_outer],
                plot_decorations=False)

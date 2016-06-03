# -*- coding: utf-8 -*-

from __future__ import print_function
import gmplot
from os.path import join
import numpy

telescope_layout = join('v5.tm', 'layout_wgs84.txt')
layout = numpy.loadtxt(telescope_layout, delimiter=',')
# station_radius_m = 35.0 / 2.0
station_radius_m = 200.0
gmap = gmplot.GoogleMapPlotter(layout[0, 1], layout[0, 0], zoom=10)
for i, (lng, lat) in enumerate(layout):
    gmap.circle(lat, lng, station_radius_m, color='r')
gmap.draw('gmap_plot.html')



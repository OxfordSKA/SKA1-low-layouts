#!venv/bin/python

import numpy
from os.path import join

coords = numpy.loadtxt(join('SKA1_low_v4a.tm', 'layout_wgs84.txt'))
print coords.shape
print numpy.mean(coords[:, 0])
print numpy.mean(coords[:, 1])

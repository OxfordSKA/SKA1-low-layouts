# -*- coding: utf-8 -*-
import numpy
import matplotlib.pyplot as pyplot

x1 = numpy.logspace(0, numpy.log10(100), 20)
y = numpy.ones(20)

fig = pyplot.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.semilogx(x1, y, '+')
ax.set_ylim(0.9, 1.1)
ax.grid()
pyplot.show()

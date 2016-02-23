# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from os.path import join
import os
import taper_function
import numpy
import matplotlib.pyplot as pyplot
import shutil


def generate_apodisation_file(station_file, sll):
    layout = numpy.loadtxt(station_file)
    x = layout[:, 0]
    y = layout[:, 1]
    w = taper_function.taylor(x, y, sll)
    print(numpy.sum(w), x.shape[0])
    pyplot.scatter(x, y, s=20, c=w)
    pyplot.show()
    numpy.savetxt('TEMP.txt', w)


if __name__ == '__main__':
    sll = -28
    model_dir = join('..', 'beam_patterns', 'models')
    models = [d for d in os.listdir(model_dir)
              if os.path.isdir(join(model_dir, d)) and
              d.endswith('.tm') and
              not '_apod' in d]

    for i, model in enumerate(models):
        model_in = join(model_dir, model)
        model_out = join(model_dir, os.path.splitext(model)[0] + '_apod.tm')
        print(i, model_in, model_out)
        if os.path.isdir(model_out):
            shutil.rmtree(model_out)
        shutil.copytree(model_in, model_out)
        stations = [d for d in os.listdir(model_out)
                    if os.path.isdir(join(model_out, d)) and
                    d.startswith('station')]
        for station in stations:
            layout_file = join(model_out, station, 'layout.txt')
            layout = numpy.loadtxt(layout_file)
            w = taper_function.taylor(layout[:, 0], layout[:, 1], sll)
            apod_file = join(model_out, station, 'apodisation.txt')
            numpy.savetxt(apod_file, w)

    # s = 0
    # station_file = join(model_dir, models[0],
    #                     'station%03i' % s, 'layout.txt')
    # generate_apodisation_file(station_file, sll)


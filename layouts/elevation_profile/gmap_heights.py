# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy
import time
from googlemaps import googlemaps
from os.path import join, exists
import math
import matplotlib.pyplot as pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle


def main():
    # TODO(BM) write ALL meta-data into pickle (eg. longitude and latitude grid)
    # TODO(BM) investigate path query API ('|' symbol) to reduce queries.
    size = 500
    layout_wgs84 = numpy.loadtxt(join('models', 'v5.tm', 'layout_wgs84.txt'),
                                 delimiter=',')
    layout_enu = numpy.loadtxt(join('models', 'v5.tm', 'layout.txt'))
    delta_x_m = layout_enu[:, 0].max() - layout_enu[:, 0].min()
    delta_y_m = layout_enu[:, 1].max() - layout_enu[:, 1].min()
    delta_max = max(delta_x_m, delta_y_m)
    num_locations = size * size
    delta_lat = layout_wgs84[:, 1].max() - layout_wgs84[:, 1].min()
    delta_lng = layout_wgs84[:, 0].max() - layout_wgs84[:, 0].min()
    pad = max(delta_lat, delta_lng) / 4.0
    lng = numpy.linspace(layout_wgs84[:, 0].min() - pad,
                         layout_wgs84[:, 0].max() + pad, size)
    lat = numpy.linspace(layout_wgs84[:, 1].min() - pad,
                         layout_wgs84[:, 1].max() + pad, size)
    lng_grid, lat_grid = numpy.meshgrid(lng, lat)
    locations = zip(lat_grid.flatten(), lng_grid.flatten())
    print(size, len(locations))

    pickle_file = join('elevation_profile',
                       'elevation_%i.pickle' % num_locations)

    if exists(pickle_file):
        with open(pickle_file, 'rb') as handle:
            elevation = pickle.load(handle)
    else:
        raise RuntimeError('arrg')
        # key_ = 'AIzaSyDQ0SeZ11PAWWX_E71nrxNJpoxEMD7WF7o'
        key_ = ''
        block_size = 250
        gmaps = googlemaps.Client(key=key_)
        blocks = int(math.ceil((size * size) / float(block_size)))
        print(size, blocks)
        # response = gmaps.elevation((layout_wgs84[0, 1], layout_wgs84[0, 0]))
        # resolution_m = response[0]['resolution'] * 10.0
        # size = int(math.ceil(delta_max / resolution_m))
        elevation = numpy.zeros(size * size)
        print('-' * 20)
        for block in range(blocks):
            start = block * block_size
            end = start + block_size
            if end > num_locations:
                block_size = num_locations - start
            print(block, 'of', blocks, ' - ', start, end, block_size)
            t0 = time.time()
            response = gmaps.elevation(locations[start:end])
            print('block %i complete %.2f s' % (block, time.time() - t0))
            for i in range(block_size):
                elevation[start + i] = response[i]['elevation']
        elevation = elevation.reshape((size, size))
        with open(pickle_file, 'wb') as handle:
            pickle.dump(elevation, handle)

    print('done')

    # mean_el = numpy.mean(elevation)
    core_el = elevation[size/2, size/2]
    elevation -= core_el
    el_range = numpy.max(numpy.abs(elevation))
    # el_min = core_el - el_range
    # el_max = core_el + el_range
    el_min = -el_range
    el_max = +el_range

    cmap = pyplot.cm.get_cmap('seismic', 2048)
    extent = [lng_grid.min(), lng_grid.max(), lat_grid.min(), lat_grid.max()]
    font_size_ = 'small'
    fig = pyplot.figure(figsize=(10, 8))
    # fig.subplots_adjust(left=0.08, bottom=0.08, right=0.93, top=0.98, hspace=0.0,
    #                     wspace=0.0)
    ax = fig.add_subplot(111)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    im = ax.imshow(elevation, interpolation='nearest', cmap=cmap,
                   origin='lower', extent=extent, vmin=el_min, vmax=el_max,
                   alpha=0.7)
    ax.plot(layout_wgs84[:, 0], layout_wgs84[:, 1], 'o', color='k', ms=3)
    t = numpy.linspace(el_min, el_max, 7)
    cbar = ax.figure.colorbar(im, cax=cax, ticks=t, format='%.2f')
    cbar.ax.tick_params(labelsize=font_size_)
    cbar.set_label('delta elevation [m]', fontsize=font_size_)
    cbar.ax.tick_params(labelsize=font_size_)
    ax.set_xlabel('longitude', fontsize=font_size_)
    ax.set_ylabel('latitude', fontsize=font_size_)
    ax.set_title('Google elevation API data for SKA1 Low site with the V5 layout')
    ax.tick_params(axis='both', which='major', labelsize=font_size_)
    ax.tick_params(axis='both', which='minor', labelsize=font_size_)
    ax.axis('tight')
    pyplot.savefig('elevations_%i.png' % num_locations)
    pyplot.close(fig)

if __name__ == '__main__':
    main()

#!/usr/bin/python
"""Calculate phase centre directions.

Usage:
    casapy --nogui --nologger --log2term -c calculate_pointing_directions.py
"""

import numpy as np
import os
import numpy.random as rand


if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # Telescope position (https://www.skatelescope.org/australia/)
    lon0 = 116.631289
    lat0 = -26.697024
    alt = 0.0
    # Datetime
    t0_year = 2016
    t0_month = 2
    t0_day = 25
    t0_hour = 10
    t0_min = 30
    t0_sec = 0
    # Elevation(s)
    azimuth = [0.0, 0.0, 0.0]
    elevation = [90.0, 67.5, 45.0]
    outfile = os.path.join('pointings.txt')
    seed = 1
    # -------------------------------------------------------------------------

    t0_utc = ('%04i/%02i/%02i/%02i:%02i:%05.2f' % (t0_year, t0_month, t0_day,
                                                   t0_hour, t0_min, t0_sec))
    epoch0 = me.epoch('UTC', t0_utc)
    t0_mjd_utc = epoch0['m0']['value']
    time0 = qa.quantity(t0_mjd_utc, unitname='d')

    fh = file(outfile, 'w')
    fh.write('# %s\n' % ('-' * 70))
    fh.write('# + %i pointings for telescope at:\n' % len(elevation))
    fh.write('#   - lon0 = %.8f deg.\n' % lon0)
    fh.write('#   - lat0 = %.8f deg.\n' % lat0)
    fh.write('# + Seed = %i\n' % seed)
    fh.write('# + Time = %s UTC\n' % t0_utc)
    fh.write('# %s\n' % ('-' * 70))
    fh.write('# RA, Dec, CASA date-time, MJD, az, el\n')

    rand.seed(seed)
    telescope_position = me.position('WGS84',
                                     qa.quantity(lon0, 'deg'),
                                     qa.quantity(lat0, 'deg'),
                                     qa.quantity(alt, 'm'))

    for az, el in zip(azimuth, elevation):
        direction0 = me.direction('AZEL', qa.quantity(az, 'deg'),
                                  qa.quantity(el, 'deg'))
        me.doframe(epoch0)
        me.doframe(telescope_position)
        direction_radec = me.measure(direction0, 'J2000')
        ra = direction_radec['m0']['value'] * (180. / np.pi)
        dec = direction_radec['m1']['value'] * (180. / np.pi)
        fh.write('% 20.15f % 20.15f %.10f % 9.2f % 9.2f\n' %
                 (ra, dec, t0_mjd_utc, az, el))
    fh.close()

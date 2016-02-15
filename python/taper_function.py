# -*- coding: utf-8 -*-
import numpy
import time


def taylor_win(n, nbar=4, sll=-30):
    """
    http://www.dsprelated.com/showcode/6.php

    from http://mathforum.org/kb/message.jspa?messageID=925929:

    A Taylor window is very similar to Chebychev weights. While Chebychev
    weights provide the tighest beamwidth for a given side-lobe level, taylor
    weights provide the least taper loss for a given sidelobe level.

    'Antenna Theory: Analysis and Design' by Constantine Balanis, 2nd
    edition, 1997, pp. 358-368, or 'Modern Antenna Design' by Thomas
    Milligan, 1985, pp.141-148.
    """
    def calculate_fm(m, sp2, a, nbar):
        n = numpy.arange(1, nbar)
        p = numpy.hstack([numpy.arange(1, m, dtype='f8'),
                          numpy.arange(m+1, nbar, dtype='f8')])
        num = numpy.prod((1 - (m**2/sp2) / (a**2 + (n - 0.5)**2)))
        den = numpy.prod(1.0 - m**2 / p**2)
        fm = ((-1)**(m + 1) * num) / (2.0 * den)
        return fm
    a = numpy.arccosh(10.0**(-sll/20.0))/numpy.pi
    sp2 = nbar**2 / (a**2 + (nbar - 0.5)**2)
    w = numpy.ones(n)
    fm = numpy.zeros(nbar)
    summation = 0
    k = numpy.arange(n)
    xi = (k - 0.5 * n + 0.5) / n
    for m in range(1, nbar):
        fm[m] = calculate_fm(m, sp2, a, nbar)
        summation = fm[m] * numpy.cos(2.0 * numpy.pi * m * xi) + summation
    w += 2.0 * summation
    return w


def taylor(x, y, sll=-28):
    x = numpy.asarray(x, dtype='f8')
    y = numpy.asarray(y, dtype='f8')
    num_antennas = x.shape[0]
    x -= (x.min() + x.max()) / 2.0
    y -= (y.min() + y.max()) / 2.0

    nbar = int(numpy.ceil(2.0 * (numpy.arccosh(10**(-sll / 20.0)) /
                                 numpy.pi)**2+0.5))
    w_taylor = taylor_win(10000, nbar, sll)
    w_taylor = w_taylor[5000:]
    r_vec = numpy.sqrt(x**2 + y**2)
    rr = numpy.linspace(0.0001, r_vec.max()+0.001, 5000)

    ab = numpy.ones(num_antennas, dtype='i8')
    for ii in range(num_antennas):
        r_dif = numpy.abs(r_vec[ii] - rr)
        bb = numpy.where(r_dif == r_dif.min())
        ab[ii] = bb[0]

    w_ch = w_taylor[ab]
    w_ch = w_ch / w_ch.max()

    return w_ch


def taylor_2(x, y, r, w_taylor, rr):
    x -= r / 2.0
    y -= r / 2.0
    r = (x**2 + y**2)**0.5

    r_diff = numpy.abs(r - rr)
    bb = numpy.where(r_diff == r_diff.min())
    ab = bb[0]
    w_ch = w_taylor[ab]
    w_ch = w_ch / w_ch.max()
    return w_ch


if __name__ == '__main__':
    t0 = time.time()
    for i in range(10000):
        w = taylor(numpy.random.random(1), numpy.random.random(1), -28)
    print('time taken = %.3f s' % (time.time() - t0))

    r = 1.0
    sll = -28
    nbar = int(numpy.ceil(2.0 * (numpy.arccosh(10**(-sll / 20.0)) /
                                 numpy.pi)**2+0.5))
    w_taylor = taylor_win(10000, nbar, sll)
    w_taylor = w_taylor[5000:]
    rr = numpy.linspace(0.0001, r+0.001, 5000)
    t0 = time.time()
    for i in range(10000):
        x = numpy.random.rand()
        y = numpy.random.rand()
        w = taylor_2(x, y, r, w_taylor, rr)
    print('time taken = %.3f s' % (time.time() - t0))

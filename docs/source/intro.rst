Introduction
============

:math:`\texttt{naif}` is a pure-python package for numerical analysis
of frequencies. It implements the NAFF algorithm introduced by Laskar
(`1990
<https://ui.adsabs.harvard.edu/abs/1990Icar...88..266L/abstract>`__,
`1999
<https://link.springer.com/chapter/10.1007/978-94-011-4673-9_13>`__)
and further developed by `Valluri & Merritt (1998)
<https://ui.adsabs.harvard.edu/abs/1998ApJ...506..686V/abstract>`__ --
see also `Price-Whelan et al (2016)
<https://ui.adsabs.harvard.edu/abs/2016MNRAS.455.1079P/abstract>`__
and their released package `SuperFreq
<https://superfreq.readthedocs.io/en/latest/>`__. It is based on the
`Fortran implementation
<https://sites.lsa.umich.edu/mvalluri/software/>`__ of `Valluri &
Merritt (1998)
<https://ui.adsabs.harvard.edu/abs/2016MNRAS.455.1079P/abstract>`__,
but it introduces a few improvements. It is also intended to be a
transparent, well documented and easy-to-use implementation.

The algorithm is summarized in Appendix A of `Beraldo e Silva et al
2023
<https://ui.adsabs.harvard.edu/abs/2023arXiv230304828B/abstract>`__. If
you use :math:`\texttt{naif}` for a publication, please cite this paper.

Using ``find_peak_freqs``
-------------------------

To start using it, we first import the relevant modules::

  >>> import numpy as np
  >>> import naif
  
Then, let's assume that you have an orbit :math:`(x, y, z, v_x, v_y,
v_z)` integrated in a given potential, where these are arrays of size
:math:`N`, as is the time :math:`t`. For simplicity, let's assume the
potential is spherical (although the frequency analysis does not
require that) and let's estimate the frequencies in the radial and
azimuthal directions, :math:`\Omega_r` and :math:`\Omega_\varphi` (the
latitudinal direction is degenerate with the azimuthal). We calculate
the relevant coordinates::

  >>> r = np.sqrt(x**2 + y**2 + z**2)
  >>> phi = np.arctan2(y, x)
  >>> Lz = (x*vy - y*vx)

where :math:`L_z` is the z-component of the angular momentum. We define the
time-series f which we use as input for the frequency
analysis. This can be the coordinate itself, e.g.::

  >>> fr = r
  
For the azimuthal component, however, it is advantageous to define the
complex array (see Papaphilippou & Laskar, `1996
<https://ui.adsabs.harvard.edu/abs/1996A%26A...307..427P/abstract>`__,
`1998
<https://ui.adsabs.harvard.edu/abs/1998A%26A...329..451P/abstract>`__
and `Beraldo e Silva et al 2023
<https://ui.adsabs.harvard.edu/abs/2023arXiv230304828B/abstract>`__)::

  >>> fphi = np.sqrt(2.*np.abs(Lz))*(np.cos(phi) + 1j*np.sin(phi))
       
Then we can extract the frequencies with larger amplitudes in the spectrum::

  >>> n_freqs = 5 # number of frequencies to extract per orbit
  >>> om_r, a_r, spec_r = naif.find_peak_freqs(fr, t, n_freqs=n_freqs, spec=True)
  >>> om_phi, a_phi, spec_phi = naif.find_peak_freqs(fphi, t, n_freqs=n_freqs, spec=True)

In case we are not interested in seeing the spectra, we simply do::

  >>> om_r, a_r = naif.find_peak_freqs(fr, t, n_freqs=n_freqs)
  >>> om_phi, a_phi = naif.find_peak_freqs(fphi, t, n_freqs=n_freqs)

And in case we just want the leading frequency (the one with largest amplitude)::
  
  >>> om_r, a_r = naif.find_peak_freqs(fr, t)
  >>> om_phi, a_phi = naif.find_peak_freqs(fphi, t)

Note that, while the leading frequency is often the fundamental
frequency of motion in the corresponding coordinate (i.e. the time
derivative of the angle variable), **this is not always the case**. What
is guaranteed is that for quasi-periodic orbits each frequency in the
spectrum is an integer combination of the fundamental frequencies, but
identifying the fundamental frequencies themselves needs to be done
judiciously.

See the tutorials for a more complete example and discussion on the
extraction of the fundamental frequencies.

Performance and reproducibility
-------------------------------

The frequency refinement and the amplitudes are computed from inner
products :math:`\langle f, e^{i\omega t}\rangle`, i.e. integrals over
the time-series evaluated with Simpson's rule. When the time-steps are
uniform (as is the case for most orbit integrators), Simpson's rule
reduces to a dot product with a fixed set of weights, which are
computed once per call (see :func:`naif.simpson_weights`). Since
version 0.1.4, this makes :func:`naif.find_peak_freqs` faster than
calling :func:`scipy.integrate.simpson` in every inner product (by a
factor of 1.1 to 1.8 in our tests, larger for shorter time-series), with the same
result up to round-off: the frequencies differ from the ones obtained
with :func:`scipy.integrate.simpson` at the level set by the tolerance
of Brent's method (``brent_tol``). If the time-steps are not uniform,
:func:`scipy.integrate.simpson` is used automatically. The behavior of
version 0.1.3 and earlier (:func:`scipy.integrate.simpson` in every
inner product) can be recovered exactly with::

  >>> om_r, a_r = naif.find_peak_freqs(fr, t, fast_simpson=False)

Note also that for **real** time-series (e.g. ``fr = r``) the spectral
lines at :math:`+\omega` and :math:`-\omega` have exactly the same
amplitude. Their relative order in the output (and, if ``n_freqs`` is
odd, which one of the pair is extracted last) is therefore set by
round-off and should not be given any meaning; for these series, only
:math:`|\omega|` is relevant. Complex time-series (e.g. ``fphi``
above) do not have this degeneracy.

A faster alternative: ``hunt_peak_freqs``
-----------------------------------------

If the time-steps are uniform, the same frequencies can be extracted
with :func:`naif.hunt_peak_freqs` (new in version 0.1.5), which has the
same inputs and outputs as :func:`naif.find_peak_freqs`::

  >>> om_r, a_r = naif.hunt_peak_freqs(fr, t, n_freqs=n_freqs)
  >>> om_phi, a_phi = naif.hunt_peak_freqs(fphi, t, n_freqs=n_freqs)

It follows the method of `Hunter (2002)
<https://ui.adsabs.harvard.edu/abs/2002SSRv..102...83H/abstract>`__:
the time-series is transformed once with an FFT, the frequency of each
line is obtained from the ratio of two adjacent DFT coefficients, and
lines are subtracted in frequency space, with no maximisation of the
spectrum and no Gram-Schmidt orthogonalisation. With the same window
parameter ``p``, it gives the same frequencies as
:func:`naif.find_peak_freqs`, and it is considerably faster. For the
isochrone orbits of the tutorials (:math:`10^4` points per orbit, 5
frequencies per time-series), it is about 8 times faster than
:func:`naif.find_peak_freqs` with the Simpson weights described above,
and about 12 times faster than :func:`naif.find_peak_freqs` in version
0.1.3 and earlier. The gain is larger when more frequencies are
extracted, since each extra line costs only a few operations in
frequency space; for the leading frequency alone, it is about 3 and 4.5
times, respectively. See the isochrone tutorial for a comparison.

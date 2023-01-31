Introduction
============

:math:`\texttt{naif}` is a pure-python package for numerical analysis of
frequencies. It implements the NAFF algorithm introduced by Laskar
(1990, 1993) and further developed by Valluri & Merritt (1998). It is
based on the Fortran implementation of Valluri & Merritt (1998), but
it introduces a few improvements. It is also intended to be a
transparent, well documented and easy-to-use implementation.

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
complex array (see Papaphilippou & Laskar, 1996, 1998 and Beraldo e Silva+ 2023)::

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

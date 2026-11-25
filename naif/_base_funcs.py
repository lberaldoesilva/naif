import numpy as np
import math
from scipy import integrate

twopi = 2.*np.pi

#--------------------
def chi_p(t, p=1):
    r""" Window function \chi_p(t)

    Parameters
    ----------
    t: float array (size N)
       Time symmetric, from -T/2 to T/2
    p: int, optional
       p parameter

    Returns
    -------
    float array
       Window function chi_p
    """
    
    T = t[-1] - t[0] # total integration time
    fact_p = math.factorial(p)
    fact_2p = math.factorial(2*p)
    return (2.**p*fact_p**2/fact_2p)*(1. + np.cos(twopi*t/T))**p
# -----------------------
def abs_fft_spectrum(f):
    """Return abs(fft(f)/N) with full FFT ordering and length N.

    If f is real, use rfft internally and reconstruct the full-length
    spectrum with the same ordering as np.fft.fft.
    This should be faster than using fft.

    If f is complex, use the usual complex FFT.
    """

    N = len(f)

    if np.isrealobj(f):
        spec_pos = np.abs(np.fft.rfft(f) / N)
        spec = np.zeros(N, dtype=float)
        n_pos = len(spec_pos)

        # Positive frequencies: 0, +df, ..., +Nyquist if N even
        spec[:n_pos] = spec_pos

        # Negative frequencies reconstructed by Hermitian symmetry
        if N % 2 == 0:
            # even N: rfft includes both zero and Nyquist;
            # neither should be duplicated
            spec[n_pos:] = spec_pos[1:-1][::-1]
        else:
            # odd N: no exact Nyquist term
            spec[n_pos:] = spec_pos[1:][::-1]
        return spec
    else:
        return np.abs(np.fft.fft(f) / N)
#-------------------------
# Simpson-rule weights for uniformly sampled time series.
#
# The Simpson rule is linear in the integrand, so for a fixed time grid it
# reduces to a dot product with a fixed weight vector w:
#     integrate.simpson(f, x=t) == sum_i w_i f_i   (up to round-off).
# Since t is the same for all the inner products of one call to
# find_peak_freqs (~20 or more), computing w once and using dot products
# avoids the per-call overhead of scipy.integrate.simpson, which dominates
# the cost of find_peak_freqs for typical series lengths.
#
# Cache of unit-step weights, keyed by the number of points N:
_UNIT_SIMPSON_WEIGHTS = {}
# Size of the small grids used to obtain the end corrections and the
# interior pattern of the rule (one odd and one even size):
_N_SMALL_ODD, _N_SMALL_EVEN = 21, 22
# Number of points at each end taken directly from the small grid
# (scipy's rule for even N modifies the last 3 weights; 8 is a safe margin):
_N_END = 8

def _unit_simpson_weights(N):
    """ Simpson weights for N points with unit spacing (cached)

    The weights reproduce scipy.integrate.simpson(f, dx=1.0) for the
    installed scipy version, including its treatment of an even number of
    points. They are built from scipy's own rule applied to the identity
    matrix of a small grid (end corrections and interior pattern), so no
    N x N matrix is needed. A consistency check against scipy is done once
    per N; if it fails, None is returned and the callers fall back to
    scipy.integrate.simpson.

    Parameters
    ----------
    N: int
       Number of points (N >= 3)

    Returns
    -------
    float array (size N) or None
       Weights w such that sum(w*f) = integrate.simpson(f, dx=1.0)
    """
    if N in _UNIT_SIMPSON_WEIGHTS:
        return _UNIT_SIMPSON_WEIGHTS[N]

    if N <= _N_SMALL_EVEN:
        # small grids: weights directly from scipy's rule
        w = integrate.simpson(np.eye(N), dx=1.0, axis=1)
    else:
        # small grid with the same parity as N:
        M = _N_SMALL_ODD if (N % 2) else _N_SMALL_EVEN
        w_M = integrate.simpson(np.eye(M), dx=1.0, axis=1)
        # interior values alternate with the parity of the index;
        # take them from the middle of the small grid, far from the ends:
        c = M // 2
        w_even = w_M[c] if (c % 2 == 0) else w_M[c + 1]
        w_odd = w_M[c] if (c % 2 == 1) else w_M[c + 1]
        w = np.where(np.arange(N) % 2 == 0, w_even, w_odd)
        # end corrections, aligned with the start and with the end:
        w[:_N_END] = w_M[:_N_END]
        w[-_N_END:] = w_M[-_N_END:]

    # consistency check against scipy with a fixed, non-trivial test vector:
    f_test = np.cos(0.37*np.arange(N)) + 0.01*np.arange(N)
    ref = integrate.simpson(f_test, dx=1.0)
    if not np.isclose(np.dot(w, f_test), ref, rtol=1e-12, atol=1e-12*np.sum(np.abs(f_test))):
        w = None

    _UNIT_SIMPSON_WEIGHTS[N] = w
    return w
# -----------------------
def simpson_weights(t, rtol=1e-8):
    """ Simpson-rule weights for a uniformly sampled time grid

    Parameters
    ----------
    t: float array (size N)
       Time-steps of the time series
    rtol: float, optional
       Relative tolerance used to decide whether the sampling is uniform
       (all time-steps equal to (t[-1] - t[0])/(N - 1) within rtol)

    Returns
    -------
    float array (size N) or None
       Weights w such that sum(w*f) equals integrate.simpson(f, x=t)
       up to round-off; None if the sampling is not uniform (or N < 3),
       in which case scipy.integrate.simpson should be used.
    """
    t = np.asarray(t)
    N = t.shape[-1]
    if (t.ndim != 1) or (N < 3):
        return None
    dt = (t[-1] - t[0])/(N - 1)
    if not np.allclose(np.diff(t), dt, rtol=rtol, atol=0.):
        return None
    w_unit = _unit_simpson_weights(N)
    if w_unit is None:
        return None
    return dt*w_unit
#-------------------------
# Scalar product with Window function:
#def inner_prod(t, u1_chi, u2):
def inner_prod(t, u1_chi, u2, axis=-1, w=None):
    """ Inner product <u_1, u_2>

    Parameters
    ----------
    t: float array (size N)
       Time symmetric, from -T/2 to T/2
    u1_chi: complex array
       u_1 * chi_p(t) - 1st arg. of inner prod. times window
    u_2: complex array
       Second argument of inner product
    axis : int, optional
        Axis along which to integrate. Default is -1 (just to vectorize the call, making it faster)
    w: float array (size N) or None, optional
        Simpson weights for the grid t (see simpson_weights).
        If given, the integral is computed as a dot product with w
        (faster, same result up to round-off); if None (default),
        scipy.integrate.simpson is used.

    Returns
    -------
    complex
       Innter product <u_1, u_2>
    """
    
    T = t[-1] - t[0]
    integrand = u1_chi*np.conj(u2)
    if w is None:
        return (1./T)*integrate.simpson(integrand, x=t, axis=axis)
    # same Simpson rule, as a dot product along the integration axis:
    return (1./T)*(np.moveaxis(integrand, axis, -1) @ w)
# -----------------------
def mn_phi_om(om, f_chi, t, w=None):
    r""" Calculates -\|(phi(omega)\| = -\|<f(t), exp(i om t)>\|

    The continuous projection of the time series onto the frequency space.
    It calcuates the minimum (instead of the maximum) 
    because Brent's looks for minima

    Parameters
    ----------
    om: float
        Frequency omega (continuous)
    f_chi: complex array
        f_k * chi_p(t) - the windowed time-series
    t: float array
        Time symmetric, from -T/2 to T/2
    w: float array or None, optional
        Simpson weights for the grid t, passed to inner_prod
        (None: use scipy.integrate.simpson)

    Returns
    -------
    float or ndarray
        -\|phi(omega)\|. Scalar if `om` is scalar, otherwise array.
    """

    #return -np.abs(inner_prod(t, f_chi, np.exp(1j*om*t)))
    om = np.asarray(om)
    # If on is scalar, same as old version:
    if om.ndim == 0:
        return -np.abs(inner_prod(t, f_chi, np.exp(1j * om * t), w=w))

    # Vectorized behavior: one row per trial frequency
    exp_om_t = np.exp(1j * om[:, None] * t[None, :])

    return -np.abs(inner_prod(t, f_chi[None, :], exp_om_t, axis=1, w=w))
# -----------------------
def gs(t, u, e, chi, w=None):
    """ Gram-Schimidt orthonomal basis
    
    For each peak identified at om, build vector 
    u = exp(i* om * t),
    and obtain e_k's normal to all previous ones
    (by Gram-Schimidt orthonomalization).

    Parameters
    ----------
    t: float array
       Time symmetric, from -T/2 to T/2
    u: complex array
       u_k = exp(i om_k t) - non-orthonormal vectors
    e: complex, array
       e_k: set of orthonormal vectors already calc.
       (it has only elements up to k)
    chi: float array
       chi_p(t) - the window function
    w: float array or None, optional
       Simpson weights for the grid t, passed to inner_prod
       (None: use scipy.integrate.simpson)

    Returns
    -------
    complex array
       Orthonormal vectors
    """
    
    k = len(e)
    u_chi = u*chi
    # projection of u_k onto e_j:
    proj_ue_kj = np.zeros((k,1), dtype=np.complex128)
    for j in range(k):
        proj_ue_kj[j] = inner_prod(t, u_chi, e[j], w=w)

    tmp_e = u - np.sum(proj_ue_kj*e, axis=0)
    tmp_e_chi = tmp_e*chi
    return tmp_e/np.sqrt(inner_prod(t, tmp_e_chi, tmp_e, w=w))

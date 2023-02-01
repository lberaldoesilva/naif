import numpy as np
from scipy import integrate

twopi = 2.*np.pi

#--------------------
def chi_p(t, p=1):
    """ Window function \chi_p(t)

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
    fact_p = np.math.factorial(p)
    fact_2p = np.math.factorial(2*p)
    return (2.**p*fact_p**2/fact_2p)*(1. + np.cos(twopi*t/T))**p
# -----------------------
# Scalar product with Window function:
def inner_prod(t, u1_chi, u2):
    """ Inner product <u_1, u_2>

    Parameters
    ----------
    t: float array (size N)
       Time symmetric, from -T/2 to T/2
    u1_chi: complex array
       u_1 * chi_p(t) - 1st arg. of inner prod. times window
    u_2: complex array
       Second argument of inner product

    Returns
    -------
    complex
       Innter product <u_1, u_2>
    """
    
    T = t[-1] - t[0]
    integrand = u1_chi*np.conj(u2)
    return (1./T)*integrate.simps(integrand, t)
# -----------------------
def mn_phi_om(om, f_chi, t):
    """ Calculates -\|(phi(omega)\| = -\|<f(t), exp(i om t)>\|

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

    Returns
    -------
    float
        -\|phi(omega)\|
    """

    return -np.abs(inner_prod(t, f_chi, np.exp(1j*om*t)))
# -----------------------
def gs(t, u, e, chi):
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
        proj_ue_kj[j] = inner_prod(t, u_chi, e[j])

    tmp_e = u - np.sum(proj_ue_kj*e, axis=0)
    tmp_e_chi = tmp_e*chi
    return tmp_e/np.sqrt(inner_prod(t, tmp_e_chi, tmp_e))

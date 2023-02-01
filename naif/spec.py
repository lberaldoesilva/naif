import numpy as np
from scipy.optimize import minimize_scalar as minimize
from ._base_funcs import inner_prod, gs, chi_p, mn_phi_om

twopi = 2.*np.pi

#--------------------
def find_peak_freqs(f_k, t, n_freqs=1, p=1, spec=False,
                    brent_tol=1e-10, eps_spec=1e-7, n_scan_peak=100):
    """Finds frequencies of peaks in the power spectrum,
    from highest to lowest amplitudes

    Parameters
    ----------
    f_k: float array (size N)
     Time-series associated with a coordinate;
     e.g.: f = r, or f = r + ivr
    t: float array (size N)
     Time-steps for the time-series
    n_freqs: int, optional
     Maximum number of frequencies to extract. 1 if only the leading frequency
    p: int, optional
         The p parameter of the Window function chi_p; p=0 for no windowing; p=1 for the Hanning window
    spec: boolean, optional
         Output the spectra before extraction of each frequency?
    brent_tol: float, optional
         Tolearance error for Brent's minimum finder method
    eps_spec: float, optional
         Minimum amplitude for keeping extracting freqss. If below that, it ends before extracting all freqs.
    n_scan_peak: int, optional
         Number of points where phi(omega) is evaluated. In case Brent's method fails (rarely used).

    Returns
    -------
    om_k: float array or number
          Extracted frequencies, in descending order of amplitude
    a_k: complex array or number
          Amplitudes associated to the frequencies om_k
    spec_k: (optional, depending on spec) complex array;
          Format (n_freqs, N) or size N. Full (windowed) spectrum before extraction of kth freq.
    """
    
    N = len(f_k)
    
    tc = 0.5*(t[0] + t[-1]) # center of the time base-line
    t_sym = t - tc # time symmetric around tc
    T = t[-1] - t[0]
    fourpi_T = 2.*twopi/T
    chi = chi_p(t_sym, p=p) # window function

    # normal vectors from Gram-Schimidt orthonomalization:
    e = np.zeros((n_freqs, N), dtype=np.complex128)
    om_k = np.zeros(n_freqs) 
    a_k = np.zeros(n_freqs, dtype=np.complex128)
    spec_k = np.zeros((n_freqs, N))
    
    # frequencies where DFT is evaluated:
    om = twopi*np.fft.fftfreq(N, T/N)
    # windowed time-series:
    f_k_chi = f_k*chi

    # Extracting "fake" line at om = 0 (sum of amplitudes):
    # Because of the windowing, this line has a finite width,
    # and needs to be extracted like the others:
    # e_0 = np.exp(1j*om[0]*t_sym)
    # because om[0] = 0:
    e_0 = np.ones_like(t_sym)
    # complex amplitude of om = 0:
    a_0 = inner_prod(t_sym, f_k_chi, e_0)
    # Extract peak from time-series:
    f_k = f_k - a_0*e_0

    # Start real extraction
    for k in range(n_freqs):
        f_k_chi = f_k*chi
        # original NAFF starts without windowing:
        # FT_k = np.fft.fft(f_k)/N
        # but windowing from the start identifies freqs.
        # in the right order, and the 1st is the leading one:
        spec_k[k] = np.abs(np.fft.fft(f_k_chi)/N) 

        # identify raw (discrete) max. and a range around it:
        om_max = om[np.argmax(spec_k[k])]
        om_inf = om_max - fourpi_T
        om_sup = om_max + fourpi_T

        # mn_phi(omega) = - |<f(t), e^(i*omega*t)>|
        # values used as brackets in Brent's method:
        mn_phi_om_max = mn_phi_om(om_max, f_k_chi, t_sym)
        mn_phi_om_inf = mn_phi_om(om_inf, f_k_chi, t_sym)
        mn_phi_om_sup = mn_phi_om(om_sup, f_k_chi, t_sym)
        
        # Normally, phi(omega) has a maximum
        # (and -phi a minimum) near om_max,
        # and both -phi(om_inf) and
        # -phi(om_sup) > -phi(omega_max):
        if ((mn_phi_om_inf > mn_phi_om_max) &
            (mn_phi_om_sup > mn_phi_om_max)):
            best_peak = minimize(mn_phi_om, args=(f_k_chi, t_sym), 
                                 bracket=(om_inf, om_max, om_sup), 
                                 tol=brent_tol, method='brent')
            # the frequency at the peak:
            om_k[k] = best_peak.x
        else:
            # when it's crowded with peaks
            # (or the interval is not large enough)
            # and the minimum is not well bracketed,
            # we scan mn_phi_om looking for the local minimum
            print ('Frequency ',k+1,
                   ' - Peak not found in first shot. Refining...')
            try:
                scan_om = np.linspace(om_inf, om_sup, n_scan_peak)
                scan_phi = np.zeros(len(scan_om))
                for i in range(n_scan_peak):
                    scan_phi[i] = mn_phi_om(scan_om[i],
                                            f_k_chi,
                                            t_sym)
                # identify among where derivative changes sign,
                # the one with minimum value:
                d1 = np.diff(scan_phi)
                d1sign = np.sign(d1)
                signchange = ((np.roll(d1sign, 1)[1:] - d1sign[1:]) != 0).astype(int)
                idx = np.where(signchange ==1)[0]
                idx_min = idx[np.argmin(scan_phi[idx])]
                om_k[k] = scan_om[idx_min]
            except:
                print ('Frequency ',k+1,
                       'Unable to find peak frequency')
                break

        #un-normalized vector:
        u = np.exp(1j*om_k[k]*t_sym)

        if (k==0):
            # The first one is necessarily normalized:
            e[k] = u
        else:
            # Gram-Schimidt ortonormalization to get e from u:
            e[k] = gs(t_sym, u, e[:k], chi)

        # complex amplitude at om_k:
        a_k[k] = inner_prod(t_sym, f_k_chi, e[k])
        # remove spectral line from time-series:
        f_k = f_k - a_k[k]*e[k]
        if (np.abs(a_k[k]) < eps_spec):
            break
    # deliver frequencies in descending order of amplitude:
    idx_sort = np.argsort(-np.abs(a_k))
    a_k = a_k[idx_sort]
    om_k = om_k[idx_sort]
    if (spec==True):
        spec_k = spec_k[idx_sort]
        if (n_freqs==1):
            out_spec = spec_k[0]
        else:
            out_spec = spec_k[:k+1]

    if (n_freqs==1):
        out_om = om_k[0]
        out_a = a_k[0]
    else:
        # :k+1 is in case less than n_freqs are extracted:
        out_om = om_k[:k+1]
        out_a = a_k[:k+1]
        
    if (spec==False):
        return out_om, out_a
    else:
        return out_om, out_a, out_spec

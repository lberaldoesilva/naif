import numpy as np
from scipy.optimize import minimize_scalar as minimize
from ._base_funcs import inner_prod, gs, chi_p, mn_phi_om, abs_fft_spectrum, simpson_weights
import warnings
from ._base_funcs import (hunter_dft, hunter_window, hunter_window_coeffs,
                          hunter_kernel, hunter_theta_brentq,
                          hunter_theta_closed, _dsinc)

twopi = 2.*np.pi

#--------------------
def find_peak_freqs(f_k, t, n_freqs=1, p=1, spec=False,
                    brent_tol=1e-10, eps_spec=1e-7, n_scan_peak=100, verbose=True,
                    fast_simpson=True):
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
    verbose: boolean, optional
         Print a message when the peak is not bracketed at the first attempt?
    fast_simpson: boolean, optional
         If True (default) and t is uniformly sampled, the Simpson-rule inner
         products are computed as dot products with precomputed weights
         (see naif.simpson_weights), which is several times faster and gives
         the same result up to round-off. If False, or if t is not uniformly
         sampled, scipy.integrate.simpson is used in every inner product.

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
    # Simpson weights for the time grid, computed once and reused in all
    # inner products of this call (None -> scipy.integrate.simpson):
    w = simpson_weights(t_sym) if fast_simpson else None

    # normal vectors from Gram-Schimidt orthonomalization:
    e = np.zeros((n_freqs, N), dtype=np.complex128)
    om_k = np.zeros(n_freqs) 
    a_k = np.zeros(n_freqs, dtype=np.complex128)

    #    spec_k = np.zeros((n_freqs, N))
    if spec:
        spec_k = np.zeros((n_freqs, N))
    else:
        spec_k = None
    
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
    a_0 = inner_prod(t_sym, f_k_chi, e_0, w=w)
    # Extract peak from time-series:
    f_k = f_k - a_0*e_0

    # Start real extraction
    for k in range(n_freqs):
        f_k_chi = f_k*chi
        # original NAFF starts without windowing:
        # FT_k = np.fft.fft(f_k)/N
        # but windowing from the start identifies freqs.
        # in the right order, and the 1st is the leading one:

        # spec_k[k] = np.abs(np.fft.fft(f_k_chi)/N)
        spec_now = abs_fft_spectrum(f_k_chi)
        #spec_now = np.abs(np.fft.fft(f_k_chi) / N)
        if spec:
            spec_k[k] = spec_now

        # identify raw (discrete) max. and a range around it:
        #om_max = om[np.argmax(spec_k[k])]
        om_max = om[np.argmax(spec_now)]
        om_inf = om_max - fourpi_T
        om_sup = om_max + fourpi_T

        # mn_phi(omega) = - |<f(t), e^(i*omega*t)>|
        # values used as brackets in Brent's method:
        mn_phi_om_max = mn_phi_om(om_max, f_k_chi, t_sym, w)
        mn_phi_om_inf = mn_phi_om(om_inf, f_k_chi, t_sym, w)
        mn_phi_om_sup = mn_phi_om(om_sup, f_k_chi, t_sym, w)
        
        # Normally, phi(omega) has a maximum
        # (and -phi a minimum) near om_max,
        # and both -phi(om_inf) and
        # -phi(om_sup) > -phi(omega_max):
        if ((mn_phi_om_inf > mn_phi_om_max) &
            (mn_phi_om_sup > mn_phi_om_max)):
            best_peak = minimize(mn_phi_om, args=(f_k_chi, t_sym, w), 
                                 bracket=(om_inf, om_max, om_sup), 
                                 tol=brent_tol, method='brent')
            # the frequency at the peak:
            om_k[k] = best_peak.x
        else:
            # when it's crowded with peaks
            # (or the interval is not large enough)
            # and the minimum is not well bracketed,
            # we scan mn_phi_om looking for the local minimum
            if verbose:
                print ('Frequency ',k+1,
                       ' - Peak not found in first shot. Refining...')
            try:
                scan_om = np.linspace(om_inf, om_sup, n_scan_peak)
                # Vectorizing mn_phi_om (faster), but we can use with scalar too:
                
                # scan_phi = np.zeros(len(scan_om))
                # for i in range(n_scan_peak):
                #     scan_phi[i] = mn_phi_om(scan_om[i], f_k_chi, t_sym)
                scan_phi = mn_phi_om(scan_om, f_k_chi, t_sym, w)
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
            e[k] = gs(t_sym, u, e[:k], chi, w=w)

        # complex amplitude at om_k:
        a_k[k] = inner_prod(t_sym, f_k_chi, e[k], w=w)
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
#-------------------------------------------------
#--------------------
def hunt_peak_freqs(f_k, t, n_freqs=1, p=1, spec=False,
                    eps_spec=1e-7, verbose=True, theta_solver='auto',
                    uniform_rtol=1e-6):
    r"""Finds frequencies of peaks in the spectrum using only DFTs
    (Hunter 2002, Space Sci. Rev. 102, 83), from highest to lowest
    amplitudes.

    Alternative to the NAFF-based naif.find_peak_freqs, with the same
    inputs and outputs. Equation numbers below refer to Hunter (2002).

    0. The time series is sampled at 2N+1 equally spaced times on
       [t_c - T, t_c + T], the two end points are averaged, and its DFT
       F_k (k = 1-N..N, frequency k pi/T) is computed with one FFT
       (eq. 6; naif._base_funcs.hunter_dft). Windowing by chi_p
       (eq. 7) is applied in frequency space with the 3-point
       recursion of eqs. (9)-(10), giving F^(p)_k
       (naif._base_funcs.hunter_window). The mean (om = 0) is removed.

    Then, for each line:

    1. the largest abs(F^(p)_k) is located;
    2. the ratio C = F^(p)_{k-1}/F^(p)_k of two adjacent coefficients
       is matched to that of a pure exponential (eq. 15),
       S^(p)(theta + pi)/S^(p)(theta) = C, where S^(p) is the windowed
       DFT of exp(i nu t) (eqs. 11-12, 36); this gives
       theta = nu T - k pi and the frequency nu = (theta + k pi)/T.
       How eq. (15) is solved is set by theta_solver;
    3. the amplitude is A = F^(p)_k/S^(p)(nu T - k pi) (eq. 16);
    4. the line is subtracted in frequency space (Sec. 3.4, text
       following eq. 16):
       F^(p)_k -> F^(p)_k - A S^(p)(nu T - k pi), for all k.

    Parameters
    ----------
    f_k: float or complex array (size N)
     Time-series associated with a coordinate;
     e.g.: f = r, or f = r + ivr
    t: float array (size N)
     Time-steps for the time-series. Must be uniformly spaced.
     If N is even, the last point is not used.
    n_freqs: int, optional
     Maximum number of frequencies to extract
    p: int, optional
     The p parameter of the window function chi_p; p=0 for no
     windowing; p=1 for the Hanning window (default, as in
     naif.find_peak_freqs). Hunter (2002) recommends 3 <= p <= 5:
     the accuracy for an isolated line increases quickly with p,
     but lines get wider (2(p+1) bins), which hurts crowded spectra.
    spec: boolean, optional
     Output abs(windowed DFT) before extraction of each frequency?
     (size 2N', with 2N' = N-1 or N-2, in np.fft order of
     frequencies k*pi/T; see naif._base_funcs.hunter_dft)
    eps_spec: float, optional
     Minimum amplitude for keeping extracting freqs.
    verbose: boolean, optional
     Print a message when the ratio method fails for a line (the
     frequency of the closest DFT bin is then used).
    theta_solver: {'auto', 'closed', 'brentq'}, optional
     How the ratio equation (eq. 15) is solved for theta:

     - 'closed': large-N closed form theta = pi [p - C(p+1)]/(1+C),
       the inverse of the large-N ratio (p pi - theta)/[theta + (p+1)pi]
       given in the caption of Fig. 2, which follows from the large-N
       kernel of eq. (14) (naif._base_funcs.hunter_theta_closed).
       For p >= 1 it is as accurate as 'brentq' (abs(delta theta)
       <~ 3e-11 at N = 500, round-off for N >= 1000) and ~1e3-1e4
       times cheaper per line.
       For p = 0 it is only accurate to abs(delta theta) ~ 0.25/N^2, and
       a warning is issued.
     - 'brentq': exact finite-N equation (eqs. 11-12, 15), whose
       unique root in -(p+1)pi < theta < p pi (Appendix, eqs. 34-37)
       is found with Brent's method
       (naif._base_funcs.hunter_theta_brentq). Accurate to ~1e-12
       for any N and p, but slower.
     - 'auto' (default): 'closed' for p >= 1, 'brentq' for p = 0.

     Here N is half the number of points in the time series, and the
     error in frequency is delta nu = delta theta/T.
    uniform_rtol: float, optional
     Relative tolerance for checking that t is uniformly spaced.

    Returns
    -------
    om_k: float array or number
          Extracted frequencies, in descending order of amplitude
    a_k: complex array or number
          Amplitudes associated to the frequencies om_k, with phases
          relative to the center of the time-series,
          t_c = (t[0] + t[-1])/2 (as in naif.find_peak_freqs)
    spec_k: (optional, depending on spec) float array
          Format (n_freqs, 2N') or size 2N'. Windowed abs(DFT) before
          extraction of kth freq.
    """
    t = np.asarray(t, dtype=float)
    dts = np.diff(t)
    if np.max(np.abs(dts - dts.mean())) > uniform_rtol*np.abs(dts.mean()):
        raise ValueError('hunt_peak_freqs requires uniformly spaced t')
    if theta_solver not in ('auto', 'closed', 'brentq'):
        raise ValueError("theta_solver must be 'auto', 'closed' or 'brentq'")
    if theta_solver == 'auto':
        theta_solver = 'brentq' if p == 0 else 'closed'
    if theta_solver == 'closed' and p == 0:
        warnings.warn("theta_solver='closed' with p=0 is only accurate to "
                      "|delta theta| ~ 0.25/N^2; use 'brentq' (or 'auto')")

    F, k, N, T, t_c = hunter_dft(f_k, t)
    Fp = hunter_window(F, p)
    n2 = 2*N
    # reference time for the output phases (same as find_peak_freqs):
    t_ref = 0.5*(t[0] + t[-1])

    # Remove the line at om = 0 (the mean), like find_peak_freqs.
    # S^(p)(-k pi) = c_k for |k| <= p, and 0 otherwise; S^(p)(0) = 1:
    c = hunter_window_coeffs(p)
    a_0 = Fp[0]
    for cl, l in zip(c, range(-p, p + 1)):
        Fp[l % n2] -= a_0*cl

    om_k = np.zeros(n_freqs)
    a_k = np.zeros(n_freqs, dtype=np.complex128)
    spec_k = np.zeros((n_freqs, n2)) if spec else None

    for j in range(n_freqs):
        absF = np.abs(Fp)
        if spec:
            spec_k[j] = absF
        i = np.argmax(absF)
        ip = (i + 1) % n2
        im = (i - 1) % n2
        # pair (k-1, k) bracketing the line: the line lies between the
        # largest coefficient and its largest neighbour
        pairs = [ip, i] if absF[ip] > absF[im] else [i, ip]
        nu = None
        for ik in pairs:
            # F^(p)_ik can be exactly zero, e.g. at k = 0 once the mean has
            # been removed (pair (-1, 0)); the ratio is undefined there.
            if Fp[ik] == 0:
                continue
            C = np.real(Fp[(ik - 1) % n2]/Fp[ik])
            if np.isfinite(C) and C > 0:
                try:
                    if theta_solver == 'closed':
                        theta = hunter_theta_closed(C, p)
                    else:
                        theta = hunter_theta_brentq(C, N, p)
                    nu = (theta + k[ik]*np.pi)/T
                    break
                except ValueError:
                    pass
        if nu is None:
            if verbose:
                print('Frequency ', j + 1,
                      ' - ratio method failed; using the DFT bin frequency')
            nu = k[i]*np.pi/T

        # amplitude from the largest coefficient:
        A = Fp[i]/hunter_kernel(nu*T - k[i]*np.pi, N, p)
        # subtract the line in frequency space. G^(p)_k(nu) for all k is
        # obtained by windowing S(nu T - k pi) with the same recursion:
        Fp = Fp - A*hunter_window(_dsinc(nu*T - k*np.pi, N), p)

        om_k[j] = nu
        a_k[j] = A*np.exp(1j*nu*(t_ref - t_c))
        if np.abs(A) < eps_spec:
            break

    # deliver frequencies in descending order of amplitude:
    n_out = j + 1
    om_k = om_k[:n_out]
    a_k = a_k[:n_out]
    idx_sort = np.argsort(-np.abs(a_k))
    om_k = om_k[idx_sort]
    a_k = a_k[idx_sort]
    if spec:
        spec_k = spec_k[:n_out][idx_sort]

    if n_freqs == 1:
        out = (om_k[0], a_k[0])
        if spec:
            out = out + (spec_k[0],)
    else:
        out = (om_k, a_k)
        if spec:
            out = out + (spec_k,)
    return out
#-------------------------------------------------

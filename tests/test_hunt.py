"""Tests for naif.hunt_peak_freqs (Hunter 2002) and its helpers."""
import numpy as np
import pytest
import naif
from naif._base_funcs import (hunter_dft, hunter_window, hunter_kernel,
                              hunter_ratio, hunter_theta_brentq,
                              hunter_theta_closed)

T_ARR = np.linspace(0., 200., 10001)
OM = np.array([1.2345678, -0.876543, 2.71828, 0.3141592])
AMP = np.array([1., 0.3*np.exp(1j), 0.05j, 0.01])


def _signal(t):
    tc = 0.5*(t[0] + t[-1])
    return sum(a*np.exp(1j*o*(t - tc)) for a, o in zip(AMP, OM))


@pytest.mark.parametrize('p', [0, 1, 3, 5])
def test_kernel_is_exact_dft_of_exponential(p):
    # windowed DFT of exp(i nu t) equals S^(p)(nu T - k pi), eqs. (10)-(12)
    t = np.linspace(-10., 10., 101)
    nu = 3.3
    F, k, N, T, t_c = hunter_dft(np.exp(1j*nu*t), t)
    assert t_c == 0.
    Fp = hunter_window(F, p)
    assert np.allclose(Fp, hunter_kernel(nu*T - k*np.pi, N, p), atol=1e-13)


@pytest.mark.parametrize('p', [0, 1, 2, 4, 8])
def test_ratio_inversion_brentq(p):
    N = 500
    for theta in np.linspace(-(p + 1)*np.pi, p*np.pi, 9)[1:-1]:
        C = hunter_ratio(theta, N, p)
        assert abs(hunter_theta_brentq(C, N, p) - theta) < 1e-10


@pytest.mark.parametrize('p', [1, 2, 4, 8])
def test_ratio_inversion_closed(p):
    # closed form is exact to round-off for p >= 1 at N >= 500
    N = 500
    for theta in np.linspace(-(p + 1)*np.pi, p*np.pi, 9)[1:-1]:
        C = hunter_ratio(theta, N, p)
        assert abs(hunter_theta_closed(C, p) - theta) < 1e-9


def test_closed_p0_error_scales_as_N_minus_2():
    theta = 0.4
    errs = [abs(hunter_theta_closed(hunter_ratio(theta, N, 0), 0) - theta)
            for N in (500, 1000)]
    assert 1e-8 < errs[0] < 1e-5
    assert 3. < errs[0]/errs[1] < 5.


@pytest.mark.parametrize('p, tol_om, tol_a', [(1, 1e-8, 1e-6), (3, 1e-12, 1e-10),
                                              (5, 1e-12, 1e-10)])
def test_multi_line_complex(p, tol_om, tol_a):
    om, a = naif.hunt_peak_freqs(_signal(T_ARR), T_ARR, n_freqs=4, p=p)
    assert np.max(np.abs(om - OM)) < tol_om
    assert np.max(np.abs(a - AMP)) < tol_a


def test_real_series_pairs_match_naff():
    f = np.real(_signal(T_ARR))
    om_d, a_d = naif.hunt_peak_freqs(f, T_ARR, n_freqs=8, p=3)
    om_n, a_n = naif.find_peak_freqs(f, T_ARR, n_freqs=8, p=3)
    # real series: each line appears as a +- pair with conjugate amplitudes
    assert np.allclose(np.sort(np.abs(om_d)), np.sort(np.abs(om_n)), rtol=1e-8)
    for o, a in zip(om_d, a_d):
        j = np.argmin(np.abs(om_d + o))
        assert abs(a_d[j] - np.conj(a)) < 1e-8


def test_spec_output_and_scalar_return():
    om, a, s = naif.hunt_peak_freqs(_signal(T_ARR), T_ARR, n_freqs=1, p=3,
                                        spec=True)
    assert np.isscalar(om) and abs(om - OM[0]) < 1e-12
    assert s.shape == (10000,)


def test_nonuniform_time_raises():
    t = np.sort(np.random.default_rng(0).uniform(0, 100, 1001))
    with pytest.raises(ValueError):
        naif.hunt_peak_freqs(np.cos(t), t)


def test_default_p_is_hanning():
    om, a = naif.hunt_peak_freqs(_signal(T_ARR), T_ARR, n_freqs=4)
    assert np.max(np.abs(om - OM)) < 1e-8


@pytest.mark.parametrize('p', [1, 3])
def test_closed_and_brentq_agree(p):
    f = _signal(T_ARR)
    om_c, a_c = naif.hunt_peak_freqs(f, T_ARR, n_freqs=4, p=p, theta_solver='closed')
    om_b, a_b = naif.hunt_peak_freqs(f, T_ARR, n_freqs=4, p=p, theta_solver='brentq')
    om_a, a_a = naif.hunt_peak_freqs(f, T_ARR, n_freqs=4, p=p)
    assert np.max(np.abs(om_c - om_b)) < 1e-12
    assert np.array_equal(om_a, om_c)


def test_auto_uses_brentq_for_p0():
    f = _signal(T_ARR)
    om_a, _ = naif.hunt_peak_freqs(f, T_ARR, n_freqs=2, p=0)
    om_b, _ = naif.hunt_peak_freqs(f, T_ARR, n_freqs=2, p=0, theta_solver='brentq')
    assert np.array_equal(om_a, om_b)


def test_closed_p0_warns():
    with pytest.warns(UserWarning):
        naif.hunt_peak_freqs(_signal(T_ARR), T_ARR, p=0, theta_solver='closed')


def test_bad_theta_solver_raises():
    with pytest.raises(ValueError):
        naif.hunt_peak_freqs(_signal(T_ARR), T_ARR, theta_solver='newton')


def test_no_divide_by_zero_at_k0():
    # Lines exactly on the bins k=-1 and k=-2 (amplitude -0.7): the leading
    # coefficient is at k=-1, the pair (k-1, k)=(-2, -1) gives a negative
    # ratio, and the next pair would divide by F^(p)_0, which is exactly zero
    # after removing the mean. The frequency falls back to the bin value.
    import warnings
    N, T = 1000, 50.
    t = np.linspace(-T, T, 2*N + 1)
    f = np.exp(-1j*np.pi*t/T) - 0.7*np.exp(-2j*np.pi*t/T)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        om, a = naif.hunt_peak_freqs(f, t, verbose=False)
    assert abs(om*T/np.pi + 1) < 1e-12

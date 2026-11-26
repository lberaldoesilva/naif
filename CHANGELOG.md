# Changelog

## Unreleased

- `hunt_peak_freqs`: no more `RuntimeWarning: divide by zero` when the
  leading line is at the lowest negative frequency (k = -1) and the
  first pair of coefficients gives no usable ratio. The next pair would
  divide by the coefficient at k = 0, which is exactly zero once the mean
  is removed; that pair is now skipped. Results are unchanged: the
  frequency still falls back to the DFT bin value, as in 0.1.5.

## 0.1.5 (2026-11-25)

### New: `naif.hunt_peak_freqs`

An alternative to `naif.find_peak_freqs` that extracts frequencies using
only discrete Fourier transforms, following Hunter (2002, Space Sci.
Rev. 102, 83):

- one FFT of the time series (Hunter's eq. 6), windowed by `chi_p` in
  frequency space with a 3-point recursion (eqs. 9-10);
- the frequency of each line from the ratio of two adjacent windowed
  coefficients (eq. 15), the amplitude from eq. (16), and line
  subtraction in frequency space. There is no maximisation of the
  spectrum and no Gram-Schmidt orthogonalisation.

Same inputs and outputs as `find_peak_freqs` (`f_k`, `t`, `n_freqs`,
`p`, `spec`, `eps_spec`, `verbose`), with default `p=1`. Differences:

- `t` must be uniformly spaced (checked with `uniform_rtol`).
- The new option `theta_solver` sets how the ratio equation is solved:
  - `'auto'` (default) uses `'closed'` for `p >= 1` and `'brentq'` for
    `p = 0`;
  - `'closed'` is the large-N closed form, theta = pi [p - C(p+1)]/(1+C),
    which inverts the large-N ratio of Hunter's Fig. 2;
  - `'brentq'` finds the root of the exact finite-N equation (Appendix)
    with Brent's method.
- No `brent_tol`, `n_scan_peak` or `fast_simpson` arguments, since they
  do not apply.

In tests on isochrone orbits (10^4 points, 5 frequencies per series)
and in logarithmic and barred potentials, `hunt_peak_freqs` gives the
same accuracy as `find_peak_freqs` at the same `p`. With 5 frequencies
per series it runs about 8 times faster than `find_peak_freqs` in 0.1.4,
and about 12 times faster than in 0.1.3; for the leading frequency
alone, about 3 and 4.5 times. At `p >= 2` it can be more accurate for well-separated
lines.

The helper functions are in `naif._base_funcs`: `hunter_dft`,
`hunter_window`, `hunter_window_coeffs`, `hunter_kernel`,
`hunter_ratio`, `hunter_theta_closed` and `hunter_theta_brentq`.

### Other changes

- `find_peak_freqs` is unchanged from 0.1.4.
- First test suite, `tests/test_hunt.py`, run with `pytest`.
- Docs:
  - `hunt_peak_freqs` and its helpers in the API reference;
  - a new section in the introduction;
  - a section in the isochrone tutorial comparing it with
    `find_peak_freqs` in frequencies and run time;
  - the parallel tutorial was rewritten. It now stores the results of
    all orbits (before, only the last block was kept), selects the
    fundamental Omega_phi, and compares with the analytical frequencies.
    It runs one task per orbit for both time-series, and compares
    `hunt_peak_freqs` and `find_peak_freqs` in parallel with
    `hunt_peak_freqs` in a serial loop.

## 0.1.4 (2026-11-25)

### Faster Simpson-rule inner products for uniform time grids

- New function `naif.simpson_weights(t)` returns Simpson-rule weights for
  uniformly sampled time grids. They reproduce `scipy.integrate.simpson`,
  including its rule for an even number of points, and are checked
  against scipy once per number of points.
- `find_peak_freqs` computes the weights once per call and passes them to
  all inner products (new optional argument `w` in `inner_prod`,
  `mn_phi_om` and `gs`). This makes frequency extraction faster, by a
  factor of 1.1 to 1.8 in later tests (larger for shorter series), with
  the same result up to round-off. Non-uniform time grids
  fall back to `scipy.integrate.simpson`.
- New option `find_peak_freqs(..., fast_simpson=True)`.
  `fast_simpson=False` reproduces version 0.1.3 exactly.
- Raw docstrings in `chi_p` and `mn_phi_om`, which avoid a
  `SyntaxWarning` for invalid escape sequences in Python >= 3.12.
- Docs: `simpson_weights` in the API reference, and a new section
  "Performance and reproducibility" in the introduction.

## 0.1.3 (2026-05-10)

- Faster `find_peak_freqs`:
  - new `naif.abs_fft_spectrum`, which uses `rfft` for real series;
  - `mn_phi_om` and `inner_prod` are vectorised over trial frequencies
    (new `axis` argument in `inner_prod`).
- New `verbose` option in `find_peak_freqs`.
- Docs: the parallel tutorial is now a notebook.

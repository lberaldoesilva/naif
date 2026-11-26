""" naif: numerical analysis and integer combinations of frequencies"""
from .spec import find_peak_freqs, hunt_peak_freqs
from ._base_funcs import chi_p, mn_phi_om, abs_fft_spectrum, simpson_weights
from . import _base_funcs

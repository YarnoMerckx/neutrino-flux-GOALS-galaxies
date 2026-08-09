"""
Shared cosmic-ray injection / transport core for a starburst nucleus,
used in Merckx, Correa, de Vries, Kotera, Privon, & van Eijndhoven 
(2023, Phys. Rev. D.) following Peretti, Blasi, Aharonian & Morlino 
(2019, MNRAS 487, 168) andits follow-up Peretti, Blasi, Aharonian, 
Morlino & Cristofari (2020,
MNRAS 493, 5880).

This module holds ONLY the physics that is common to both the neutrino
output (analytic_neutrino_flux.py) and the gamma-ray
output (gamma_flux.py): proton injection Q(p), the steady-state momentum
distribution f(p), and the timescales that go into it (wind advection,
hadronic losses, diffusive escape).
"""
import numpy as np
from astropy import units as u
from astropy import constants as const

# ===============================
#         INJECTION SECTION
# ===============================

def cross_section(E):
    """Inelastic p-p cross section (mb), Kelner, Aharonian & Bugayov
    (2006). Returns 0 below the pion-production threshold E_th."""
    E = np.atleast_1d(E).astype(float)
    L = np.log(E / 1e3)
    E_th = 1.22
    with np.errstate(invalid="ignore"):
        val = (34.3 + 1.88*L + 0.25*L**2) * (1 - (E_th / E)**4)**2
    return np.where(E >= E_th, val, 0.0)


def E_CR(RSN):
    """Total CR energy injection rate from SN explosions (GeV/s)."""
    E_SN_erg = 1e51 * u.erg
    E_SN_GeV = (E_SN_erg.to(u.eV)).value * 1e-9
    xi = 0.10
    R_sn_yr = RSN / u.yr
    R_sn_s = R_sn_yr.to(1 / u.s).value
    return R_sn_s * E_SN_GeV * xi


def I(p, alpha, p_max):
    mp = 0.938
    return (4 * np.pi) * p**2 * (p / mp)**(-alpha) * (np.sqrt(p**2 + mp**2) - mp) * np.exp(-p / p_max)


_Int_cache = {}

def _get_Int(alpha, plow, pup, pmax):
    key = (alpha, plow, pup, pmax)
    if key not in _Int_cache:
        mom = np.logspace(np.log10(plow), np.log10(pup), 100000)
        _Int_cache[key] = np.trapezoid(I(mom, alpha, pmax), mom)
    return _Int_cache[key]


def Qp(p, R, h, plow, pup, alpha, pmax, RSN):
    """Differential CR proton injection rate Q(p), normalized so that
    its integral over the SBN volume matches E_CR(RSN)."""
    mp = 0.938
    R_cm = (R * u.pc).to(u.cm).value
    h_cm = (h * u.pc).to(u.cm).value
    if h == 0:
        V_SBN = (4 / 3) * np.pi * R_cm**3
    else:
        V_SBN = 2 * np.pi * R_cm**2 * h_cm
    Int = _get_Int(alpha, plow, pup, pmax)
    N = E_CR(RSN) / Int
    return (N / V_SBN) * (p / mp)**(-alpha) * np.exp(-p / pmax)

# ===============================
#         TIMESCALES
# ===============================

def loss_time(p, nism):
    """Hadronic (p-p) energy-loss time (s)."""
    E = np.sqrt(p**2 + 0.938**2)
    eta = 0.5
    n_m = (nism * u.cm**-3).to(u.m**-3).value
    sigma = cross_section(E) * 1e-31
    return 1 / (eta * n_m * sigma * const.c.value)


def tau_wind(R, v, h):
    """Advective wind escape time (s)."""
    v_wind = v * 1000
    if h == 0:
        R_SBN = (R * u.pc).to(u.m).value
        return R_SBN / v_wind
    else:
        h_m = (h * u.pc).to(u.m).value
        return h_m / v_wind


def larmor(p, B):
    """Larmor radius (m)."""
    E = np.sqrt(p**2 + 0.938**2)
    B_T = (B * u.G * 1e-6).to(u.T).value
    return 3.3 * (E / B_T)


_W0_cache = {}

def W_0_trapz(k_0, d):
    key = (k_0, d)
    if key not in _W0_cache:
        integral = lambda k: k**(-d)
        logaxis = np.logspace(0, 10, 100000)
        I_ = np.trapezoid(integral(logaxis), logaxis)
        _W0_cache[key] = (k_0**d * I_)**-1
    return _W0_cache[key]


def F(k, k_0, d):
    return k * W_0_trapz(k_0, d) * (k / k_0)**(-d)


def D(E, k_0, B, d):
    """Diffusion coefficient (pc^2/s), quasi-linear theory."""
    c = const.c.value
    k_m = 1 / (larmor(E, B) * u.m)
    k_pc = k_m.to(1 / u.pc).value
    D_m2_s = (larmor(E, B) * c) / (3 * F(k_pc, k_0, d))
    D_pc2_s = (D_m2_s * u.m**2 / u.s).to(u.pc**2 / u.s).value
    return D_pc2_s


def tau_diff_quasi(p, R, h):
    """Diffusive escape time (s). Escape length L = R for a sphere
    (h == 0) or H for a cylindrical superbubble slab (h != 0) -- the
    same geometry convention already used by tau_wind."""
    L = R if h == 0 else h
    return L**2 / D(p, 1, 250, 5 / 3)


def tau_lifetime(R, vwind, p, nism, h):
    """Total CR proton lifetime (s), combining wind advection, hadronic
    losses, and diffusive escape."""
    return 1 / (
        1 / tau_wind(R, vwind, h) +
        1 / loss_time(p, nism) +
        1 / tau_diff_quasi(p, R, h)
    )

# ===============================
#     MOMENTUM DISTRIBUTION
# ===============================

def f_p(p, R, v, nism, h, plow, pup, alpha, pmax, RSN):
    """Steady-state CR proton momentum distribution f(p)."""
    return tau_lifetime(R, v, p, nism, h) * Qp(p, R, h, plow, pup, alpha, pmax, RSN)
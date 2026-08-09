"""
Secondary gamma-ray emission model for a starburst nucleus, following
Peretti, Blasi, Aharonian & Morlino (2019, MNRAS 487, 168) and its
follow-up Peretti, Blasi, Aharonian, Morlino & Cristofari (2020, MNRAS
493, 5880), with pion-decay spectra from Kelner, Aharonian & Bugayov
(2006, Phys. Rev. D 74, 034018).

Proton injection/transport (Q(p), f(p)) is shared with the neutrino
pipeline and lives in `cr_transport.py`; this module only adds the
gamma-ray-specific pieces:

    - Fgamma() / q_gamma() : the pion-decay photon spectrum and the
      resulting gamma-ray source function, stitched across a smooth
      low-/high-energy transition (see q_gamma docstring).
    - internal_absorption() : pair production on the source's own dense
      FIR/optical photon field (Peretti+2020; see also de Cea del Pozo,
      Torres & Rodriguez Marrero 2009, ApJ 698, 1054). Relevant already
      at ~1-10 TeV for typical starburst nuclei.
    - ebl_absorption() : pair production on the diffuse extragalactic
      background light over the cosmological path to the observer, via
      gammapy. Relevant at the highest energies and/or largest distances.

Both absorption effects are physically distinct and are applied
multiplicatively in Flux_gamma().
"""
import numpy as np
from astropy import units as u
from astropy import constants as const
from scipy.interpolate import PchipInterpolator

from helper_functions.cr_transport import cross_section, f_p
from helper_functions.imf_calibrations import L_IR_from_RSN

# ---------------------------------------------------------------
# EBL absorption (gammapy). Import is optional at module load time:
# if gammapy / its EBL data tables aren't available in the current
# environment, Flux_gamma still works as long as you don't pass z.
# ---------------------------------------------------------------
try:
    from gammapy.modeling.models import EBLAbsorptionNormSpectralModel
    _GAMMAPY_AVAILABLE = True
except ImportError:
    _GAMMAPY_AVAILABLE = False

_ebl_model_cache = {}

def ebl_absorption(E_gamma, z, ebl_model="franceschini17"):
    """
    EBL attenuation factor exp(-tau_EBL(E, z)) from gammapy's built-in
    tables (Franceschini+2017 by default; also available: 'dominguez11',
    'finke10', 'franceschini08', 'saldana-lopez21', etc.).

    E_gamma : energy or array of energies, in GeV.
    z       : redshift of the source.
    """
    if not _GAMMAPY_AVAILABLE:
        raise ImportError(
            "gammapy is not available in this environment -- install it "
            "and set GAMMAPY_DATA to your local EBL data path to use "
            "ebl_absorption()."
        )
    key = (ebl_model, z)
    if key not in _ebl_model_cache:
        _ebl_model_cache[key] = EBLAbsorptionNormSpectralModel.read_builtin(
            ebl_model, redshift=z
        )
    model = _ebl_model_cache[key]
    E = np.atleast_1d(E_gamma) * u.GeV
    return model(E.to(u.TeV)).value

# ===============================
#     GAMMA-RAY (PION-DECAY) DISTRIBUTION
# ===============================

def Fgamma(x, Ep):
    L = np.log(Ep / 1e3)
    Bgamma = 1.30 + 0.14*L + 0.011*L**2
    betagamma = 1 / (1.79 + 0.11*L + 0.008*L**2)
    kgamma = 1 / (0.801 + 0.049*L + 0.014*L**2)
    if x == 1:
        return 0
    else:
        first = (np.log(x) / x)
        second = ((1 - x**betagamma) / (1 + kgamma * x**betagamma * (1 - x**betagamma)))**4
        third = (1/np.log(x)) - \
                ((4 * betagamma * x**betagamma) / (1 - x**betagamma)) - \
                ((4 * kgamma * betagamma * x**betagamma * (1 - 2 * x**betagamma)) / (1 + kgamma * x**betagamma * (1 - x**betagamma)))
        return Bgamma * first * second * third
Fgamma = np.vectorize(Fgamma)

# ===============================
#         GAMMA-RAY SOURCE FUNCTION
# ===============================

def _smoothstep(t):
    """
    Standard cubic smoothstep on t in [0,1]: 3t^2 - 2t^3.
    w(0)=0, w(1)=1, and w'(0)=w'(1)=0 -- i.e. it blends in with zero
    slope at both ends, so gluing w*A + (1-w)*B to pure A (t<0) and
    pure B (t>1) is continuous in both value AND derivative, unlike a
    hard switch or a single-point-matched normalization.
    """
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _q_gamma_low(E_eval, R, v, nism, H, gammasn, pmax, RSN, cut, c=3e10):
    """
    Pion-decay delta-function/kinematic-edge branch using momentum substitution
    p_pi = sqrt(E_pi^2 - m_pi^2) to remove the 1/sqrt(E_pi^2 - m_pi^2) threshold singularity.
    """
    m_pi = 0.135  # GeV

    # 1. High-energy Kelner branch at `cut` (5 GeV) for normalization n_tilde
    x_cut = np.logspace(-4, 0, 1000)
    x_cut = x_cut[x_cut < cut / 1.22]  # Ensure proton energy E_p >= 1.22 GeV
    if len(x_cut) > 0:
        p_prot = np.sqrt((cut / x_cut)**2 - 0.938**2)
        integrand = (
            Fgamma(x_cut, cut / x_cut)
            * cross_section(cut / x_cut)
            * 1e-27
            * (1 / x_cut)
            * 4 * np.pi * p_prot**2
            * f_p(p_prot, R, v, nism, H, 0.1, 1e9, gammasn, pmax, RSN)
        )
        b = c * nism * np.trapezoid(integrand, x_cut)
    else:
        b = 0.0

    K_pi = 0.17

    # Helper function to integrate over pion momentum p_pi instead of E_pi
    def _integrate_q_pi(E_gamma_val):
        Emin = E_gamma_val + m_pi**2 / (4 * E_gamma_val)
        p_min = np.sqrt(max(0.0, Emin**2 - m_pi**2))
        p_max = min(pmax * 10.0, 1e8)

        # Construct momentum grid: handle p_min = 0 cleanly
        if p_min == 0:
            p_grid = np.concatenate([[0.0], np.geomspace(1e-5, p_max, 2000)])
        else:
            p_grid = np.geomspace(p_min, p_max, 2000)

        E_pi = np.sqrt(p_grid**2 + m_pi**2)
        p_prot = np.sqrt((0.938 + E_pi / K_pi)**2 - 0.938**2)

        q_pi = (
            c * nism / K_pi
            * cross_section(0.938 + E_pi / K_pi)
            * 1e-27
            * 4 * np.pi * p_prot**2
            * f_p(p_prot, R, v, nism, H, 0.1, 1e9, gammasn, pmax, RSN)
        )

        # dE_pi / sqrt(E_pi^2 - m_pi^2) = dp_pi / E_pi (strictly finite!)
        integrand = q_pi / E_pi
        return np.trapezoid(integrand, p_grid)

    # Compute normalization factor n_tilde
    a = 2 * _integrate_q_pi(cut)
    n_tilde = b / a if a > 0 else 1.0

    # Compute physical gamma-ray emissivity at E_eval
    return 2 * n_tilde * _integrate_q_pi(E_eval)


def _q_gamma_high(E_eval, R, v, nism, H, gammasn, pmax, RSN, c=3e10):
    """Smooth Kelner et al. (2006) branch, evaluated at E_eval directly."""
    x = np.logspace(-4, 0, 1000)
    x = x[x < E_eval / 0.938]
    p = np.sqrt((E_eval / x)**2 - 0.938**2)
    integrand = Fgamma(x, E_eval / x) * cross_section(E_eval / x) * 1e-27 * (1 / x) * \
                4 * np.pi * p**2 * f_p(p, R, v, nism, H, 0.1, 1e9, gammasn, pmax, RSN)
    I_ = np.trapezoid(integrand, x)
    return c * nism * I_


def q_gamma(E_gamma, R, v, nism, H, gammasn, pmax, RSN,
            cut_lo=5.0, cut_hi=15.0, n_samples=6):
    if E_gamma <= cut_lo:
        return _q_gamma_low(E_gamma, R, v, nism, H, gammasn, pmax, RSN, cut=cut_lo)
    elif E_gamma >= cut_hi:
        return _q_gamma_high(E_gamma, R, v, nism, H, gammasn, pmax, RSN)
    else:
        E_lo_side = np.linspace(cut_lo * 0.5, cut_lo, n_samples)
        E_hi_side = np.linspace(cut_hi, cut_hi * 2.0, n_samples)
        E_samples = np.concatenate([E_lo_side, E_hi_side])
        q_samples = np.concatenate([
            [_q_gamma_low(E, R, v, nism, H, gammasn, pmax, RSN, cut=cut_lo) for E in E_lo_side],
            [_q_gamma_high(E, R, v, nism, H, gammasn, pmax, RSN) for E in E_hi_side]
        ])
        pchip = PchipInterpolator(np.log(E_samples), np.log(q_samples))
        return np.exp(pchip(np.log(E_gamma)))


q_gamma = np.vectorize(q_gamma)

# ===============================
#   INTERNAL GAMMA-GAMMA ABSORPTION
#   (pair production on the source's own FIR/optical photon field)
# ===============================
#
# Follows Peretti, Blasi, Aharonian & Morlino
# (2019, MNRAS 487, 168): the SBN's own dust-reprocessed IR (+ optical/UV)
# radiation field acts as a target for e+e- pair production on gamma-rays
# still inside the source, distinct from cosmological EBL absorption.
#
#   eta_gg(E) = Int  n(eps) * sigma_gg(E, eps) d(eps)      [cm^-1]
#   tau_gg(E) = eta_gg(E) * L                               [dimensionless]
#
# where n(eps) is the internal photon number density per unit energy
# (graybody spectrum normalized to the radiation energy density U_rad),
# sigma_gg is the standard Gould & Schreder (1967) pair-production cross
# section, and L is the photon escape path length (R for a sphere, H for
# a cylindrical superbubble slab -- the same "L" convention now used by
# tau_wind/tau_diff_quasi in cr_transport.py).

m_e_GeV = (const.m_e * const.c**2).to(u.GeV).value       # 5.10999e-4 GeV
sigma_T_cm2 = const.sigma_T.to(u.cm**2).value              # 6.6524e-25 cm^2
k_B_GeV_per_K = (const.k_B).to(u.GeV / u.K).value           # 8.61733e-14 GeV/K
erg_to_GeV = (1 * u.erg).to(u.GeV).value


def U_rad_from_LIR(L_IR_erg_s, R_pc):
    """
    Radiation energy density (GeV/cm^3) implied by an IR luminosity
    L_IR [erg/s] escaping a region of characteristic radius R_pc [pc],
    via U_rad ~ L_IR / (4 pi R^2 c).
    """
    R_cm = (R_pc * u.pc).to(u.cm).value
    c_cms = const.c.to(u.cm / u.s).value
    U_erg_cm3 = L_IR_erg_s / (4 * np.pi * R_cm**2 * c_cms)
    return U_erg_cm3 * erg_to_GeV  # GeV/cm^3


def graybody_n(eps_GeV, U_rad_GeV_cm3, T_K):
    """
    Photon number density per unit energy, n(eps) [cm^-3 GeV^-1], for a
    blackbody/graybody spectrum at temperature T_K, normalized so that
    Int_0^inf n(eps) * eps d(eps) = U_rad_GeV_cm3.

    Normalization is done ANALYTICALLY via the exact Planck relation
      Int_0^inf x^3/(e^x - 1) dx = pi^4/15
    rather than by numerically integrating over whatever eps_grid happens
    to be passed in. This matters because eta_gg() calls this function on
    an E_gamma-dependent, often truncated grid (eps_min = m_e^2/E_gamma)
    -- normalizing on that truncated grid instead of the true full
    spectrum would silently underestimate the total radiated energy (and
    so overestimate n(eps) and hence tau_gg) right around the
    pair-production threshold, exactly where this feature matters most.
    """
    eps_GeV = np.atleast_1d(eps_GeV).astype(float)
    kT = k_B_GeV_per_K * T_K
    x = eps_GeV / kT
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        shape = np.where(x < 700, eps_GeV**2 / np.expm1(x), 0.0)
    # Exact normalization: Int n(eps) eps d(eps) = U_rad
    #   => (norm) * kT^4 * (pi^4/15) = U_rad  [since shape*eps ~ eps^3/(e^x-1)
    #      and d(eps) = kT dx integrates to kT^4 * Int x^3/(e^x-1) dx]
    norm_integral_analytic = kT**4 * (np.pi**4 / 15.0)
    if norm_integral_analytic <= 0:
        return np.zeros_like(eps_GeV)
    return (U_rad_GeV_cm3 / norm_integral_analytic) * shape


def sigma_gg(E_gamma_GeV, eps_GeV):
    """
    Gould & Schreder (1967) pair-production cross section sigma_gg(E,eps)
    [cm^2], for a gamma-ray of energy E_gamma_GeV against a target photon
    of energy eps_GeV (isotropic-field / head-on approximation)
    """
    s = E_gamma_GeV * eps_GeV
    thresh = m_e_GeV**2
    with np.errstate(invalid="ignore"):
        beta2 = 1 - thresh / s
    beta2 = np.where(s > thresh, beta2, 0.0)
    beta = np.sqrt(np.clip(beta2, 0.0, 1.0 - 1e-12))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.log((1 + beta) / (1 - beta))
    sigma = (3 * sigma_T_cm2 / 16) * (1 - beta2) * (
        2 * beta * (beta2 - 2) + (3 - beta**4) * log_term
    )
    return np.where(s > thresh, np.clip(sigma, 0.0, None), 0.0)


def eta_gg(E_gamma_GeV, U_rad_GeV_cm3, T_K, n_eps=800):
    """
    Absorption coefficient eta_gg(E) = Int n(eps) sigma_gg(E,eps) d(eps),
    in cm^-1, for a single scalar E_gamma_GeV.
    """
    kT = k_B_GeV_per_K * T_K
    eps_min = max(m_e_GeV**2 / E_gamma_GeV, kT * 1e-4)
    eps_max = kT * 60.0
    if eps_max <= eps_min:
        return 0.0
    eps_grid = np.logspace(np.log10(eps_min), np.log10(eps_max), n_eps)
    n_eps_vals = graybody_n(eps_grid, U_rad_GeV_cm3, T_K)
    integrand = n_eps_vals * sigma_gg(E_gamma_GeV, eps_grid)
    return np.trapezoid(integrand, eps_grid)


eta_gg = np.vectorize(eta_gg, excluded=["U_rad_GeV_cm3", "T_K", "n_eps"])


def tau_gg_internal(E_gamma_GeV, L_pc, U_rad_GeV_cm3, T_K):
    """
    Internal gamma-gamma optical depth tau_gg(E) = eta_gg(E) * L, with L
    [pc] the characteristic photon escape path length (R for a sphere,
    H for a cylindrical superbubble slab -- the same "L" already used in
    tau_wind/tau_diff_quasi).
    """
    L_cm = (L_pc * u.pc).to(u.cm).value
    return eta_gg(E_gamma_GeV, U_rad_GeV_cm3, T_K) * L_cm


def internal_absorption(E_gamma_GeV, L_pc, U_rad_GeV_cm3, T_K,
                         escape_model="volume"):
    """
    Suppression factor applied to the internally-produced gamma-ray flux
    due to pair production on the source's own FIR/optical photon field.

    escape_model:
      'volume' (default) -- photons are produced throughout the emitting
          volume; escape probability (1 - exp(-tau)) / tau (-> 1 as
          tau -> 0), following the radiative-transfer treatment used in
          Peretti et al. (2020).
      'shell'  -- photons are all produced near the center and must
          traverse the full path length; simple exp(-tau).
    """
    tau = tau_gg_internal(E_gamma_GeV, L_pc, U_rad_GeV_cm3, T_K)
    tau = np.atleast_1d(tau).astype(float)
    if escape_model == "shell":
        return np.exp(-tau)
    # 'volume': escape-probability form, safe at tau -> 0
    small = tau < 1e-3
    out = np.empty_like(tau)
    out[small] = 1.0 - tau[small] / 2.0  # Taylor expansion, avoids 0/0
    out[~small] = (1 - np.exp(-tau[~small])) / tau[~small]
    return out

# ===============================
#         OBSERVED FLUX (gamma-rays)
# ===============================

def Flux_gamma(E_gamma, R, v, nism, H, gammasn, pmax, RSN, D_L, z=None,
               ebl_model="franceschini17",
               internal_abs=True, L_IR=None, T_dust=40.0,
               escape_model="volume",
               calib=None,
               SN_per_Msun=1 / 100, L_IR_per_SFR_erg_s=2.5e43):
    """
    z : if given, multiplies the flux by the EXTERNAL, cosmological EBL
        attenuation factor exp(-tau_EBL(E_gamma, z)) using gammapy's
        built-in tables (see ebl_absorption() above). If None (default),
        no EBL absorption is applied.

    internal_abs : if True (default), also applies INTERNAL gamma-gamma
        absorption on the source's own FIR/optical photon field (see
        internal_absorption() above). This is a separate, physically
        distinct effect from EBL absorption and is typically far more
        important at the energies where these sources become optically
        thick (often already ~1-10 TeV), long before EBL matters.

    L_IR : internal FIR luminosity of the source [erg/s]. If None
        (default), it is estimated self-consistently from RSN via
        L_IR_from_RSN() -- pass an explicit value if you have a measured
        L_IR for your source instead (e.g. straight from the `log(LIR)`
        column already in this repo's GOALS dataframe).

    calib : if L_IR is None, this is forwarded to L_IR_from_RSN(). Leave
        as None to use the generic SN_per_Msun / L_IR_per_SFR_erg_s
        scaling, or set to "Murphy" / "Yarno NK" / "Yarno TH" to instead
        derive L_IR as the exact inverse of one of the IMF-dependent
        SNr_IMF() calibrations.

    T_dust : characteristic dust/graybody temperature [K] for the
        internal radiation field (default 40 K, typical of starburst
        nuclei; adjust to your source).

    escape_model : 'volume' or 'shell', see internal_absorption().
    """
    DL_cm = (D_L * 1e6 * u.pc).to(u.cm).value
    R_cm = (R * u.pc).to(u.cm).value
    H_cm = (H * u.pc).to(u.cm).value
    if H == 0:
        V = (4 / 3) * np.pi * R_cm**3
    else:
        V = 2 * np.pi * R_cm**2 * H_cm
    scaled_flux = (V / (4 * np.pi * DL_cm**2)) * E_gamma**2 * \
                  q_gamma(E_gamma, R, v, nism, H, gammasn, pmax, RSN)

    if internal_abs:
        if L_IR is None:
            L_IR = L_IR_from_RSN(RSN, calib=calib,
                                  SN_per_Msun=SN_per_Msun,
                                  L_IR_per_SFR_erg_s=L_IR_per_SFR_erg_s)
        L_path_pc = R if H == 0 else H
        U_rad = U_rad_from_LIR(L_IR, R)  # GeV/cm^3, always sized on R
        abs_factor = internal_absorption(E_gamma, L_path_pc, U_rad, T_dust,
                                          escape_model=escape_model)
        scaled_flux = scaled_flux * abs_factor

    if z is not None:
        scaled_flux = scaled_flux * ebl_absorption(E_gamma, z, ebl_model)

    # Match the output shape to the input E_gamma shape (e.g. a scalar in
    # gives a plain scalar out) -- internal_absorption()/ebl_absorption()
    # both go through np.atleast_1d internally, so without this a scalar
    # E_gamma would come back as a size-1 array instead of a float.
    return np.asarray(scaled_flux).reshape(np.shape(np.asarray(E_gamma, dtype=float)))

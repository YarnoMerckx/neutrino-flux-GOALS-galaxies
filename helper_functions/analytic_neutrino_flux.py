"""
Neutrino output for the starburst CR transport model.

As of the gamma-ray extension, the CR injection/transport physics
(cross_section, Qp, timescales, f_p) has moved to `cr_transport.py`.
"""
import numpy as np
from astropy import units as u
from astropy import constants as const

from helper_functions.cr_transport import (
    cross_section, E_CR, I, Qp, loss_time, tau_wind, larmor,
    W_0_trapz, F, D, tau_diff_quasi, tau_lifetime, f_p,
)

# ===============================
#     NEUTRINO DISTRIBUTION
# ===============================

def Fmu1(x, Ep):
    if x <= 0.427:
        L = np.log(Ep / 1e3)
        y = x / 0.427
        B = 1.75 + 0.204*L + 0.010*L**2
        beta = 1 / (1.67 + 0.111*L + 0.0038*L**2)
        k = 1.07 - 0.086*L + 0.002*L**2
        first = B * (np.log(y) / y) * ((1 - y**beta) / (1 + k * y**beta * (1 - y**beta)))**4
        second = (1 / np.log(y)) - \
                 (4 * beta * y**beta / (1 - y**beta)) - \
                 (4 * k * beta * y**beta * (1 - 2 * y**beta) / (1 + k * y**beta * (1 - y**beta)))
        return first * second
    else:
        return 0

Fmu1 = np.vectorize(Fmu1)

def Fe(x, Ep):
    L = np.log(Ep / 1e3)
    Be = 1 / (69.5 + 2.65*L + 0.3*L**2)
    betae = (0.201 + 0.062*L + 0.00042*L**2)**(-1/4)
    ke = (0.279 + 0.141*L + 0.0172*L**2) / (0.3 + (2.3 + L)**2)
    first = (1 + ke * np.log(x)**2)**3 / (x * (1 + 0.3 / x**betae))
    second = (-np.log(x))**5
    return Be * first * second

Fe = np.vectorize(Fe)

def Ftot(x, Ep):
    return 2 * Fe(x, Ep) + Fmu1(x, Ep)

Ftot = np.vectorize(Ftot)

# ===============================
#         SOURCE FUNCTION q
# ===============================

def q(E_nu, R, v, nism, H, gammasn, pmax, RSN):
    x = np.logspace(-4, 0, 1000)
    c = 3e10
    x = x[x < E_nu / 0.938]
    p = np.sqrt((E_nu / x)**2 - 0.938**2)
    integrand = Ftot(x, E_nu / x) * cross_section(E_nu / x) * 1e-27 * (1 / x) * \
                4 * np.pi * p**2 * f_p(p, R, v, nism, H, 0.1, 1e9, gammasn, pmax, RSN)
    I_ = np.trapezoid(integrand, x)
    return c * nism * I_
q = np.vectorize(q)

# ===============================
#         OBSERVED FLUX (neutrinos)
# ===============================

def Flux(E_nu, R, v, nism, H, gammasn, pmax, RSN, D_L):
    DL_cm = (D_L * 1e6 * u.pc).to(u.cm).value
    R_cm = (R * u.pc).to(u.cm).value
    H_cm = (H * u.pc).to(u.cm).value
    if H == 0:
        V = (4 / 3) * np.pi * R_cm**3
    else:
        V = 2 * np.pi * R_cm**2 * H_cm
    # Factor 1/3 accounts for neutrino oscillation into 3 flavors
    scaled_flux = (1 / 3) * (V / (4 * np.pi * DL_cm**2)) * E_nu**2 * \
                  q(E_nu, R, v, nism, H, gammasn, pmax, RSN)
    return scaled_flux
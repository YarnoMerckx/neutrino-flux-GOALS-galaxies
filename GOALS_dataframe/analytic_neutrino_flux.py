# ==== IMPORTS ====
#General imports
import matplotlib.pyplot as plt
import numpy as np 

#Astropy imports
from astropy import units as u 
from astropy import constants as const

#Scipy imports
from scipy.integrate import quad

# ===============================
#         INJECTION SECTION
# ===============================

# Inelastic cross-section for proton-proton collisions (in mb)
def cross_section(E):
    L = np.log(E / 1e3)  # E in MeV
    E_th = 1.22  # Threshold energy (GeV)
    return (34.3 + 1.88*L + 0.25*L**2) * (1 - (E_th / E)**4)**2

# Total CR energy injection rate from SN explosions (GeV s-1)
def E_CR(RSN):
    E_SN_erg = 1e51 * u.erg                # Typical supernova energy
    E_SN_GeV = (E_SN_erg.to(u.eV)).value * 1e-9  # Convert to GeV
    xi = 0.10                               # Acceleration efficiency
    R_sn_yr = RSN / u.yr                   # SN rate in yr-1
    R_sn_s = R_sn_yr.to(1 / u.s).value     # Convert rate to s-1
    return R_sn_s * E_SN_GeV * xi          # Total energy per second

# Source spectrum integrand: includes momentum spectrum, energy, and exponential cutoff
def I(p, alpha, p_max):
    mp = 0.938  # Proton mass in GeV/c^2
    return (4 * np.pi) * p**2 * (p / mp)**(-alpha) * (np.sqrt(p**2 + mp**2) - mp) * np.exp(-p / p_max)

# Differential CR injection rate Q(p)
def Qp(p, R, h, plow, pup, alpha, pmax, RSN):
    mp = 0.938  # Proton mass
    R_cm = (R * u.pc).to(u.cm).value       # Radius in cm
    h_cm = (h * u.pc).to(u.cm).value       # Height in cm

    # Volume calculation
    if h == 0:
        V_SBN = (4 / 3) * np.pi * R_cm**3  # Spherical volume
    else:
        V_SBN = 2 * np.pi * R_cm**2 * h_cm # Disk volume

    # Momentum integration for normalization
    mom = np.logspace(np.log10(plow), np.log10(pup), 100000)
    Int = np.trapz(I(mom, alpha, pmax), mom)
    N = E_CR(RSN) / Int                    # Normalization constant

    return (N / V_SBN) * (p / mp)**(-alpha) * np.exp(-p / pmax)

# ===============================
#         TIMESCALES
# ===============================

# Energy loss time due to hadronic collisions (in seconds)
def loss_time(p, nism):
    E = np.sqrt(p**2 + 0.938**2)
    eta = 0.5                              # Inelasticity
    n_m = (nism * u.cm**-3).to(u.m**-3).value
    sigma = cross_section(E) * 1e-31       # Convert mb to m²
    return 1 / (eta * n_m * sigma * const.c.value)

# Wind escape time (in seconds)
def tau_wind(R, v, h):
    v_wind = v * 1000                      # Convert km s-1 to m s-1
    if h == 0:
        R_SBN = (R * u.pc).to(u.m).value
        return R_SBN / v_wind
    else:
        h_m = (h * u.pc).to(u.m).value
        return h_m / v_wind

# Larmor radius (in meters)
def larmor(p, B):
    E = np.sqrt(p**2 + 0.938**2)
    B_T = (B * u.G * 1e-6).to(u.T).value   # Convert μG to Tesla
    return 3.3 * (E / B_T)                 # Approximate formula

# Trapezoidal integration for turbulence normalization constant W₀
def W_0_trapz(k_0, d):
    integral = lambda k: k**(-d)
    logaxis = np.logspace(0, 10, 100000)
    I = np.trapz(integral(logaxis), logaxis)
    return (k_0**d * I)**-1

# Turbulence spectrum function F(k)
def F(k, k_0, d):
    return k * W_0_trapz(k_0, d) * (k / k_0)**(-d)

# Diffusion coefficient in pc²/s
def D(E, k_0, B, d):
    c = const.c.value
    k_m = 1 / (larmor(E, B) * u.m)                         # Convert to m-1
    k_pc = k_m.to(1 / u.pc).value                          # Convert to m-1
    D_m2_s = (larmor(E, B) * c) / (3 * F(k_pc, k_0, d))    # m²/s
    D_pc2_s = (D_m2_s * u.m**2 / u.s).to(u.pc**2 / u.s).value
    return D_pc2_s

# Diffusion time (quasi-linear theory, in seconds)
def tau_diff_quasi(p, R):
    return R**2 / D(p, 1, 250, 5 / 3)  # B = 250 μG

# Total CR lifetime including all loss processes
def tau_lifetime(R, vwind, p, nism, h):
    return 1 / (
        1 / tau_wind(R, vwind, h) +
        1 / loss_time(p, nism) +
        1 / tau_diff_quasi(p, R)
    )

# ===============================
#     MOMENTUM DISTRIBUTION
# ===============================

# Steady-state CR momentum distribution function f(p)
def f_p(p, R, v, nism, h, plow, pup, alpha, pmax, RSN):
    return tau_lifetime(R, v, p, nism, h) * Qp(p, R, h, plow, pup, alpha, pmax, RSN)

# ===============================
#     NEUTRINO DISTRIBUTION
# ===============================

# Muon neutrino distribution from pion decay (Kelner et al. 2006)
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

Fmu1_vec = np.vectorize(Fmu1)

# Electron neutrino distribution
def Fe(x, Ep):
    L = np.log(Ep / 1e3)
    Be = 1 / (69.5 + 2.65*L + 0.3*L**2)
    betae = (0.201 + 0.062*L + 0.00042*L**2)**(-1/4)
    ke = (0.279 + 0.141*L + 0.0172*L**2) / (0.3 + (2.3 + L)**2)
    first = (1 + ke * np.log(x)**2)**3 / (x * (1 + 0.3 / x**betae))
    second = (-np.log(x))**5
    return Be * first * second

Fe = np.vectorize(Fe)

# Total neutrino spectrum (2e + 1μ)
def Ftot(x, Ep): 
    return 2 * Fe(x, Ep) + Fmu1(x, Ep)

Ftot = np.vectorize(Ftot)

# ===============================
#         SOURCE FUNCTION q
# ===============================

# Neutrino source function q(Eν)
def q(E_nu, R, v, nism, H, gammasn, pmax, RSN):
    x = np.logspace(-4, 0, 1000)
    c = 3e10  # Speed of light in cm s-1
    p = np.sqrt((E_nu / x)**2 - 0.938**2)
    integrand = Ftot(x, E_nu / x) * cross_section(E_nu / x) * 1e-27 * (1 / x) * \
                4 * np.pi * p**2 * f_p(p, R, v, nism, H, 0.1, 1e9, gammasn, pmax, RSN)
    I = np.trapz(integrand, x)
    return c * nism * I  # Units: GeV-1 cm-3 s-2

q = np.vectorize(q)

# ===============================
#     OBSERVED NEUTRINO FLUX
# ===============================

# Energy-squared scaled neutrino flux at Earth
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
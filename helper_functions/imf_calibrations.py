"""
IMF-dependent supernova-rate <-> infrared-luminosity calibrations.
    'Murphy'   -- Murphy et al. (2011)-type calibration, core-collapse SN
                  rate from 33 GHz/FIR-derived SFR, Kroupa-like IMF.
    'Yarno NK' -- alternate calibration (non-Kroupa / "NK" IMF variant).
    'Yarno TH' -- alternate calibration ("TH" IMF variant).
"""
import numpy as np

# 1 Lsun in erg/s: (Lsun in Watts) * (1e7 erg/s per Watt)
LSUN_ERG_S = 3.828e26 * 1e7  # = 3.828e33 erg/s

# ---------------------------------------------------------------
# Each entry gives the coefficient in:
#     RSN [yr^-1] = COEFF(calib) * L_IR [erg/s]
#
# IMPORTANT: these coefficients (e.g. Murphy's 3.88e-44) are already
# calibrated for L_IR in erg/s (this is exactly the Murphy et al. 2011
# 33 GHz/FIR SFR-L_TIR[erg/s] relation, divided by ~86.3 Msun/SN to
# convert SFR -> core-collapse SN rate).
# ---------------------------------------------------------------
_CALIB_COEFFS_PER_ERG_S = {
    "Murphy":   (1 / 86.3) * 3.88e-44,
    "Yarno NK": 5.973941e-46,
    "Yarno TH": 4.151310e-46,
}


def SNr_IMF(LIR_Lsun, calib):
    """
    Core-collapse supernova rate RSN [yr^-1] implied by an infrared
    luminosity LIR [Lsun], for one of several IMF-dependent calibrations
    of the SN-rate/L_IR relation:

        calib = "Murphy"    -- Murphy et al. (2011)-type calibration
        calib = "Yarno NK"  -- alternate ("NK") IMF calibration
        calib = "Yarno TH"  -- alternate ("TH") IMF calibration

    LIR_Lsun : IR luminosity in solar luminosities (Lsun). Internally
        converted to erg/s (via LSUN_ERG_S) before applying the
        erg/s-native calibration coefficient.

    This is the exact inverse relation used by L_IR_from_RSN(..., calib=...)
    below -- i.e. SNr_IMF(L_IR_from_RSN(RSN, calib=c) / LSUN_ERG_S, c) == RSN.
    """
    if calib not in _CALIB_COEFFS_PER_ERG_S:
        raise ValueError(
            f"Unknown calib {calib!r}; choose from {list(_CALIB_COEFFS_PER_ERG_S)}"
        )
    LIR_erg_s = LIR_Lsun * LSUN_ERG_S
    return _CALIB_COEFFS_PER_ERG_S[calib] * LIR_erg_s


def L_IR_from_RSN(RSN, calib=None, SN_per_Msun=1 / 100, L_IR_per_SFR_erg_s=2.5e43):
    """
    Estimate the FIR luminosity (erg/s) that accompanies a given
    supernova rate RSN [yr^-1].

    Two modes, selected by `calib`:

    1) calib=None (default) -- generic, hand-tunable two-step scaling:
         SFR [Msun/yr]  = RSN / SN_per_Msun
         L_IR [erg/s]   = SFR * L_IR_per_SFR_erg_s
       Defaults: 1 SN per 100 Msun of star formation (order-of-magnitude,
       IMF-dependent -- adjust SN_per_Msun to match the SN-rate/SFR
       relation you actually adopt elsewhere in your model, e.g. tied to
       the IMF you used for E_CR/xi), and a Kennicutt (1998)-type
       L_IR-SFR conversion (SFR = 1.7e-10 L_IR/Lsun, i.e.
       L_IR/erg/s ~ 2.2e43 * SFR/(Msun/yr); rounded here to 2.5e43 to
       allow for some infrared excess).

    2) calib="Murphy" / "Yarno NK" / "Yarno TH" -- exact algebraic
       inverse of SNr_IMF() for that calibration, i.e.
         L_IR [erg/s] = RSN / COEFF_ERG_S(calib)
       Use this when you want internal consistency with whichever
       SNr_IMF() calibration you're using elsewhere in your model
       (e.g. if you separately quote an IMF-calibrated SN rate from a
       measured L_IR and want the round trip to be exact).

    Returns L_IR in erg/s either way. This is only meant to give a
    self-consistent, order-of-magnitude default for the internal photon
    field when you don't want to specify L_IR by hand -- pass L_IR
    explicitly to internal_absorption()/Flux_gamma() if you have a
    measured value for your source (as is usually preferable for real
    sources like M82, NGC253, Arp220, etc.)
    """
    if calib is None:
        SFR = RSN / SN_per_Msun
        return SFR * L_IR_per_SFR_erg_s

    if calib not in _CALIB_COEFFS_PER_ERG_S:
        raise ValueError(
            f"Unknown calib {calib!r}; choose from "
            f"{list(_CALIB_COEFFS_PER_ERG_S)} or None"
        )
    return RSN / _CALIB_COEFFS_PER_ERG_S[calib]
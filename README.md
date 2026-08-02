# High-Energy Neutrino Emission from GOALS Galaxies

This repository models and predicts high-energy neutrino fluxes from galaxies in the **Great Observatories All-Sky LIRG Survey ([GOALS](https://goals.ipac.caltech.edu/))**.

---

## 📌 Overview

This project compiles electromagnetic data across the GOALS sample, evaluates starburst-driven neutrino fluxes using analytical models, and generates both individual point-source and diffuse flux predictions. Future updates will incorporate gamma-ray implementations.

---

## 📓 Notebook Workflow

1. **`Generate_dataframe.ipynb`**
   * Extracts electromagnetic data from the `data/split-lir/` directory for each GOALS source.
   * Calculates starburst-driven neutrino fluxes using `helper_functions.analytic_neutrino_flux`.
   * Outputs the compiled dataset used by subsequent prediction notebooks.

2. **`GOALS_flux_predictions.ipynb`**
   * Generates per-source neutrino flux predictions across the sample.

3. **`GOALS_diffuse_predictions.ipynb`**
   * Computes the total diffuse neutrino flux predictions.

4. **`NGC1068_flux_evidence.ipynb`**
   * Compares theoretical fluxes generated via `analytic_neutrino_flux.py` against IceCube's high-energy point-source observations for NGC 1068 specifically (`NGC1068_evidence_flux.txt`).

---

## 📚 References & Publications

* **Theoretical Foundations & Analytic Flux Derivation:**
  * *Phys. Rev. D* 108 (2023) 2, 023015 — [arXiv:2304.01020](https://arxiv.org/abs/2304.01020)
* **IceCube NGC 1068 Observational Data:**
  * *Science* 378 (2022) 6619, 538–543 — [arXiv:2211.09972](https://arxiv.org/abs/2211.09972)
* **GOALS Survey Details:**
  * [GOALS Project Website](https://goals.ipac.caltech.edu/)

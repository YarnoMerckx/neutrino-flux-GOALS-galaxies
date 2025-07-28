# High-energy neutrino emission from GOALS galaxies

The `GOALS_dataframe` folder contains a notebook that constructs a dataframe compiling electromagnetic data for each source in the Great Observatories All-Sky LIRG Survey ([GOALS](https://goals.ipac.caltech.edu/)), extracted from the `SplitIR` folder. It also includes starburst-driven neutrino fluxes calculated using `analytic_neutrino_flux.py`. Theoretical motivations and derivations for the analytic neutrino flux can be found in our work: Phys.Rev.D 108 (2023) 2, 023015 ([arXiv:2211.09972](https://arxiv.org/abs/2304.01020))).
 
The resulting dataframe is utilized in the notebooks `GOALS_flux_predictions.ipynb` and `GOALS_diffuse_predictions.ipynb` to generate per-source and diffuse neutrino flux predictions, respectively.

Additionally, the notebook `NGC1068_flux_evidence.ipynb` compares fluxes obtained via the script `analytic_neutrino_flux.py` with the high-energy point-source flux observed by IceCube, as reported in `NGC1068_evidence_flux.txt` and obtained from Science 378 (2022) 6619, 538–543 ([arXiv:2211.09972](https://arxiv.org/abs/2211.09972)).
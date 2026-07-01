# Fos: Semi-non-negative matrix factorization for Fos imaging data

This library contains models for analyzing Fos imaging data. It is a fork of the
Linderman-lab [`fos`](https://github.com/lindermanlab/fos) package, adapted for the
whole-brain opioid c-Fos study (Ishii et al., 2025).

## Install

```bash
pip install git+https://github.com/kenjp1223/fos
```

Requires `jax` / `jaxlib` (and, for the driver below, `dask`, `zarr`, `scikit-image`,
`tifffile`, `scipy`, `pandas`).

## Reproducible pipeline — spatial semi-NMF factorization

`notebooks/fos_counts_clean.ipynb` is the cleaned, shareable driver that produces the
**spatial clustering results** used by the manuscript's Figure 6 (and the `factor{i}.npy`
consumed by Figures 8 / S16).

Pipeline:

1. Load the per-subject whole-brain c-Fos count heatmaps (`OP_cFos_heatmap_array`), the
   subject order (`OP_cFos_fnamelist.npy`), the meta table (`OP_meta.csv`), and the Kim
   reference atlas.
2. Downsample the heatmaps from 20×20×50 µm to 50×50×50 µm.
3. Fit a Poisson semi-NMF model (`fos.seminmf_counts_2.fit_poisson_seminmf`) with **22
   factors** and **sparsity penalty 1e-2** — values selected by cross-validated model
   selection (sparsity ∈ {1e-4…1e0}, factors ∈ {8…24}; held-out log-likelihood on random
   spatial masks; the sweep was tracked with Weights & Biases).
4. Save `params.pkl`, `params.mat` (`factors`, `count_loadings`, row/col effects),
   `npy/factor{k}.npy` (upsampled to the full-res atlas grid), and `factors.zarr`.

Paths resolve from environment variables so the notebook is shareable:

- `OPIOID_DATA_ROOT` — the Figshare deposit root (expects `01_main_cfos_morphine/` and
  `shared/atlas/`).
- `OPIOID_FACTOR_RESULTS` — where to write the outputs (defaults to a timestamped folder
  under `01_main_cfos_morphine/spatial_clustering_results/`).

The downstream factor statistics and spatial-panel figures are **not** in this repo; they
live in the manuscript analysis repo (`Fig6_semiNMF.py`), which reads the outputs above.

## Repo layout

- `fos/` — the package. The model used for the manuscript is `seminmf_counts_2.py`
  (`initialize_nnsvd`, `fit_poisson_seminmf`), with helpers in `prox.py` and `utils.py`.
  The other `seminmf_*` modules are earlier/experimental variants kept for reference.
- `notebooks/fos_counts_clean.ipynb` — the reproducible driver (above).
- `scripts/` — batch/cluster scripts.

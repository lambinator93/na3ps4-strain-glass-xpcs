# Strain-Glass Dynamics in the Superionic Conductor Na3PS4

Analysis code for Bragg X-ray photon correlation spectroscopy (XPCS) measurements
reported in:

E. S. Lamb et al., "Strain-Glass Dynamics in the Superionic Conductor Na3PS4,"
arXiv:2611.04547 (2026).

Includes two-time correlation analysis, KWW fitting, avalanche detection, and
speckle-size analysis.

Raw data are available as described in the paper's Data Availability statement.

## Setup

Requires [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

This creates `.venv/` with the pinned dependencies from `uv.lock` (Python ≥ 3.13),
including `ipykernel`. Open the notebooks with any Jupyter front end using that
environment, for example:

```bash
uv run --with jupyterlab jupyter lab
```

or select `.venv` as the kernel in VS Code.

## Configuration

All paths are set in `configs/config_NPS_2_data_090825.json`:

| Key | Meaning |
| --- | --- |
| `Base` | Root directory of the raw beamtime data. Edit this to your local copy. |
| `<scan>` | Sub-directory of each scan under `Base`. |
| `Processed`, `<scan>_processed` | Folder and file name of the pre-processed HDF5 for each scan. |
| `Results`, `<scan>_results*` | Folder and file name of the XPCS results HDF5 for each scan. |
| `<scan>_temperature` | Sample temperature (K). |
| `Figures` | Output directory for all saved figures and tables. Relative paths are resolved from the repository root (the parent of `configs/`); the default `figures/` writes to `./figures/`. An absolute path also works. |

## Workflow

1. `pre_processing.ipynb` (once per scan) → processed HDF5
2. `xpcs.ipynb` (once per scan) → results HDF5
3. The remaining notebooks read those files and make the figures.

## Notebooks

Cells that call `fig.savefig` produce the publication figures. Every figure is written
to the `Figures` directory.

| Notebook | Description | Saved figures |
| --- | --- | --- |
| `pre_processing.ipynb` | Loads one raw scan, removes hot pixels, fits the Bragg peak (1D and 2D Gaussian), crops the ROI, checks peak stability, computes the Q-map and writes the processed HDF5 (plus a GIF of the speckle pattern). | `detector_bragg_powder_ring.pdf`; diagnostic PNGs in `<Figures>/<scan>/` |
| `xpcs.ipynb` | For one scan: constant-intensity elliptical ring masks, multi-tau g2 per ring, KWW fits (τ, β, A), two-time correlation and waterfalls; writes the results HDF5. | none (diagnostic plots only) |
| `results_compile.ipynb` | Compiles all BM-cubic temperatures: refits g2 of the central ring with per-scan lag cutoffs to give τ_r, β and A₀ vs. T, and plots the two-time maps and example detector frames. | `twotime.pdf`, `tau_beta.pdf`, `det_frame_{1,2,3,450,900}.pdf`, `det_frame_colorbar.pdf` |
| `q_dependence.ipynb` | g2(Q, τ) for every ring and temperature with KWW fits; τ_r and β vs. ΔQ. | `g2_q_dependence.pdf`, `tau_beta_dependence.pdf` |
| `tau_beta_errors.ipynb` | Uncertainties on τ_r, β and contrast from fits to non-overlapping age blocks of the two-time (N = 3). The last cell repeats this for N = 5. | `tau_beta_block_fits_n3.pdf`, `tau_beta_errors_n3.pdf`, `error_blocks_scan219.pdf`, table `tau_beta_errors_n3.csv` (`*_n5` files from the last cell) |
| `ttc_lineouts.ipynb` | Scan 219 (298 K): anti-diagonal cuts of the two-time at several sample ages, KWW fits vs. age, and the oscillatory residual (quasi-period from the fit, residual-peak spacing and FFT). | `ttc_219.pdf`, `isf_219.pdf`, `tau_beta_219.pdf` |
| `avalanche.ipynb` | Avalanche detection from the two-time correlation; cumulative event counts, size distribution and power-law exponent, mean magnitude, rate and maximum vs. T. | `avalanche_detection.pdf`, `cumulative_avalanche_events.pdf`, `avalanche_size_distribution.pdf`, `avalanche_size_distribution_power.pdf`, `average_avalanche_magnitude_vs_temperature.pdf`, `avalanche_rate_vs_temperature.pdf`, `avalanche_maximum_vs_temperature.pdf` |
| `waterfalls.ipynb` | Intensity waterfalls (central detector column vs. time) for the tetragonal and BM-cubic phases. | `waterfall_65.pdf`, `waterfall_combined.pdf` |
| `xrd.ipynb` | Radial I(\|Q\|) from the summed detector images, pseudo-Voigt fits, peak position Q₀ and Scherrer correlation length ξ vs. T. | `radial_iq_pseudo_voigt_allT.pdf`, `peak_position_vs_temperature_allT.pdf`, `correlation_length_vs_temperature_allT.pdf`, `radial_iq_pseudo_voigt.pdf`, `peak_position_vs_temperature.pdf`, `correlation_length_vs_temperature.pdf` |
| `q_center_of_mass.ipynb` | Intensity-weighted center of mass of the Bragg peak in Q and in detector pixels vs. time for each temperature. | `q_center_of_mass_evolution.pdf`, `q_center_of_mass_evolution_pixels.pdf` |
| `speckle_size.ipynb` | Speckle size from the autocorrelation of a grid of ROIs and the corresponding illuminated-particle size; saves the FWHM grid next to the scan data. | `roi_grid.pdf`, `central_roi.pdf` |
| `alumina.ipynb` | Laboratory powder XRD of alumina, cubic and tetragonal Na3PS4 and Stycast, with a Scherrer estimate of ξ. | `alumina_powder_pattern.pdf`, `na3ps4_cubic_tet.pdf` |

## Source (`src/`)

- `xpcs.py`: fit models (KWW, Gaussian, pseudo-Voigt), Q-maps, binning, ring masks, two-time line cuts, plotting helpers
- `autocorrelations.py`: multi-tau g2 and two-time correlation
- `avalanche_analysis.py`: avalanche detection from two-time / difference matrices
- `plotting.py`: tick and colorbar formatting

## License

MIT; see `LICENSE`.

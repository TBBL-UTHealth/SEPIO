# SEPIO

Code accompanying the publication: **"Optimization of electrode placement and information capacity for local field potentials in cortex"**.

This repository contains two complementary workflows:
1. Ordered, reproducible research scripts and notebooks under `scripts/` (used to generate results for the paper).
2. A standalone graphical interface under `GUI/` for running Multi-Device Placement Optimization (MDPO) interactively.

---
## Key Concepts

- **SEPIO**: Sparse Electrode Placement for Intracranial Optimization. Implements Monte-Carlo sensor ranking using SSPOC (Sparse Sensor Placement Optimization for Classification) in `scripts/modules/SEPIO.py`.
- **MDPO**: Multi-Device Placement Optimization. A framework of optimization approaches for jointly placing heterogeneous devices (e.g. SEEG, DISC, ECoG) to maximize voltage, SNR, or information capacity.
- **Lead Fields**: Electrode sensitivity volumes exported from Ansys Electronics Desktop (AEDT), or other electronic modelling software, and transformed for downstream voltage/SNR/IC calculations.

---
## Features

- **Lead Field Export & Import**: Export from ANSYS Maxwell via `LeadFieldExporter`, then ingest `.npz` archives using `FieldImporter`.
- **Device Optimization**: `mdpo_project.py` provides a serializable `Project` class with constraint-aware optimization (depth, proximity, angle).
- **Sensor Ranking / Classification**: `sepio.mc_train` performs Monte-Carlo SSPOC training, returning per-electrode coefficients and accuracy curves across sensor counts.
- **Multi-Device Support**: Unified handling of multiple device types with per-type noise, bandwidth, voxel scale, angle limits, depth ranges, and proximity limits.
- **Information Metrics**: Voltage, SNR, and Information Capacity (IC = bandwidth * log2(1 + SNR)) scoring inside optimization loops.
- **Visualization**:
  - Matplotlib/Seaborn plots for fitness history & electrode layouts.
  - 3D scatter of optimized devices vs ROI points.
  - Optional Open3D interactive view of best solution.
  - Disc/ECoG electrode plotting utilities in `disc_plotter.py`.
- **Assessment Artifacts**: GUI can save plot images and export best solution CSV.

---

## Repository Structure

`/scripts` contains the sequential methodology used in the paper. `/GUI` wraps core logic into a tabbed application so users can configure projects, run optimizations, and assess results without editing code.

```
GUI/
  MDPO_GUI.py          # Unified Tkinter interface (project setup, optimization, assessment)
  mdpo_project.py      # Project dataclass + optimization core for GUI
scripts/
  0_environment_setup.ipynb     # Environment and dependency setup
  1_Field_Export_From_AEDT.ipynb# Exporting fields from Ansys AEDT
  2_Polar_plots.ipynb           # Fig. 5; Electrode layout / polar visualization
  3_IC_device_separation.ipynb  # Fig. 6; Device-wise IC contribution analyses
  4_Montage_angle.ipynb         # Fig. 7; Montage / orientation effect exploration
  5_IC_sulcus.ipynb             # Fig. 8; Sulcus-focused IC investigation
  6_MDPO_visualize.py           # Fig. 9, 10, 11; Advanced visualization helpers
  7_MDPO_multidev.py            # Fig. 10; Original multi-device optimization script (legacy)
  8_MDPO_assess.ipynb           # Fig. 11, 12; Assessment & post-optimization metrics
  9_SEPIO_phantom.ipynb         # Fig. 14; Phantom + simulated validation scenario
  10_SEPIO_sulcus.ipynb         # Fig. 13; SEPIO sulcus-specific experiments
  modules/
	 SEPIO.py                # Monte-Carlo SSPOC electrode ranking
	 leadfield_importer.py   # Load & shape G matrices (FieldImporter)
	 leadfield_exporter.py   # ANSYS Maxwell export utilities (LeadFieldExporter)
	 field_metrics.py        # Field diversity & derived metrics (FieldEvaluator)
	 disc_plotter.py         # DISC/ECoG electrode plotting helpers
```



---
## Installation

1. Create and activate a virtual environment (recommended):
	```powershell
	python -m venv .venv
	.venv\Scripts\activate
	```
2. Install required Python packages:
	```powershell
	pip install -r requirements.txt
	```
3. If performing GUI assessments or 3D visualization, ensure an environment that supports Tkinter (standard on most Python distributions) and OpenGL drivers for Open3D.

---
## GUI Quick Start

1. Launch the interface:
	```powershell
	python GUI/MDPO_GUI.py
	```
2. In the "Project Setup" tab:
	- Select brain surface `.mat` file.
	- Add one or more ROI `.mat` files (labels default to `ans`).
	- Add lead field `.npz` archives (one per device type or geometry).
	- Specify device counts (comma-separated per type), names, noise (µV), bandwidth (Hz), voxel scale (mm), and optional angle limits.
	- (Optional) Click "Refresh Constraint Fields" to enter per-type depth & proximity limits.
	- Click "Create / Update Project".
3. See `/GUI/README.md` for details on each tab.

All saved plots and exports are tracked in the project's `assessment_artifacts` list.

---
## Data Access

All datasets are available via Zenodo:

[![DOI: 10.5281/zenodo.16782866](https://zenodo.org/badge/DOI/10.5281/zenodo.16782866.svg)](https://doi.org/10.5281/zenodo.16782866)

Download and extract the ZIP archive; then point the GUI or scripts to the local paths for:
- Whole brain surface `.mat`
- ROI surface `.mat` files (one or more regions)
- Lead field archives `.npz` (each containing G-matrix + metadata)


---
## Citation

If you use this code or the provided datasets, please cite the associated publication, optionally including the GitHub and Zenodo repositories.

- Optimization of electrode placement and information capacity for local field potentials in cortex — DOI: [10.1101/2025.04.25.650658](https://www.biorxiv.org/content/10.1101/2025.04.25.650658v3)
- Associated Zenodo dataset — DOI: [10.5281/zenodo.16782866](https://doi.org/10.5281/zenodo.16782866)

---
## License

See `LICENSE` for terms of use. We enccourage collaboration.

---
## Contributing & Issues

Feel free to open issues for desired improvements or modifications.

Pull requests should keep changes modular - avoid major refactors inside legacy numbered scripts. Ideally avoid them altogether.


---
## Questions

For usage questions or clarification on adapting the optimization to new device types, please open an issue or discussion thread.

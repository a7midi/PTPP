# PTPP PRL Reproducibility Package (Standalone)

This repository is a **standalone** reproducibility package for the PRL Letter and Supplemental Material contained in:

- `paper/main.tex` (compiled: `paper/main.pdf`)
- `paper/supplement.tex` (compiled: `paper/supplement.pdf`)

It includes:

1. The **simulation engine** for layered directed photonic meshes in an incoherent intensity-transport model.
2. The **condensation-DAG geometry diagnostic** used to compute the defect marker \(\Delta a^*\).
3. Scripts to **regenerate the datasets** used in the Letter.
4. Scripts to **regenerate all figures** in the Letter and Supplemental Material.

No external repositories are required.

---

## Repository layout

- `ptpp/` – glue code + sweep runners that generate the CSV datasets.
- `scdc_oproc/` – transport-engine implementation vendored into this repo (mesh generation, simulation, metrics).
- `emergence_v3/` – condensation-DAG geometry diagnostics vendored into this repo.
- `configs/` – YAML configs for the runs used in the paper.
- `results/` – **processed** CSV datasets used by the plotting scripts.
- `paper/figs/` – generated figure PDFs.
- `paper/scripts/` – figure-generation scripts.

---

## Environment setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## Quick reproduction (figures only; uses included CSVs)

This regenerates all figure PDFs in `paper/figs/` **from the included processed CSVs**.

### Main-text figures

```bash
python paper/scripts/make_main_figures.py \
  --w20_csv   results/joint_sweep_full/joint_sweep_long.csv \
  --size2_csv results/joint_sweep_size2/joint_sweep_long.csv \
  --pknot_csv results/pknot_sweep/pknot_sweep_long.csv \
  --jitter_csv results/jitter_0to50/jitter_long.csv \
  --dict_csv  results/joint_sweep_full/dictionary_global.csv \
  --out_dir   paper/figs
```

### Supplemental figures (classifier metrics + bias checks)

```bash
python paper/scripts/make_referee_response_figs.py \
  --w20_csv   results/joint_sweep_full/joint_sweep_long.csv \
  --size2_csv results/joint_sweep_size2/joint_sweep_long.csv \
  --jitter_csv results/jitter_0to50/jitter_long.csv \
  --out_dir   paper/figs
```

**Important:** the W=20 main-text selection uses **knot-off injected at channel 0** and **knot-on injected at channel 9**. Therefore `--w20_csv` must point to the *full* W=20 sweep (`results/joint_sweep_full/joint_sweep_long.csv`), which contains both injection sites.

---

## Build the PDFs locally (optional)

If you have a LaTeX distribution installed:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode supplement.tex
```

---

## Full regeneration (datasets + figures)

The commands below regenerate the processed datasets into `results/`, then rebuild the figures.

### Run A: Joint sweep (W=20, L=50)

```bash
python -m ptpp.run_joint_sweep --config configs/joint_sweep_full.yaml
```

### Run C: Size replicate with fixed interior injection (W=28, L=70)

```bash
python -m ptpp.run_joint_sweep --config configs/joint_sweep_size2.yaml
```

### Run D: Knot-strength sweep (continuous control knob)

```bash
python -m ptpp.run_pknot_sweep --config configs/pknot_sweep.yaml
```

### Run A (timing): 0–50 ps jitter sweep

```bash
python -m ptpp.run_jitter_sweep --config configs/jitter_0to50.yaml
```

### Rebuild the binned dictionary table (optional)

If you regenerate `results/joint_sweep_full/joint_sweep_long.csv`, you can also regenerate the binned dictionary table:

```bash
python -m ptpp.make_dictionary \
  --joint_sweep_csv results/joint_sweep_full/joint_sweep_long.csv \
  --out_csv results/joint_sweep_full/dictionary_global.csv
```

### Rebuild all figures

Run the **Quick reproduction** commands again to regenerate all PDFs in `paper/figs/`.

---

## Notes for reproducibility

- All statistical results in the paper are computed from **topology-only** geometric diagnostics and **incoherent** transport observables.
- Randomness is controlled by explicit integer seeds recorded in the CSV files.
- The plotting scripts do not call any simulation code; they operate only on the processed CSV artifacts.

---

## Contact

Ahmed Alayar (Independent Researcher, Kuwait)

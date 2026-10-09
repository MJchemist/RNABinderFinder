# RNABinderFinder

Predicts RNA-binding residues in proteins using an RBF-kernel SVM trained on
six structural/sequence features plus ESM-2 protein-language-model
embeddings, reduced via PCA.

Validated on a 24-protein held-out test set: **MCC = 0.620, ROC-AUC = 0.943,
PR-AUC = 0.653**.

## Repository layout

```
RNABinderFinder/
├── README.md
├── requirements.txt
├── predict_rbp.py              # run this to get predictions on new proteins
├── color_predictions.py        # optional: visualize predictions in PyMOL
├── model.joblib                 # trained model
├── rbp_artifacts/
│   ├── pca300.joblib           # fitted PCA-300 transform
│   └── propensity_raw.json     # amino-acid RNA-binding propensity lookup
└── example/
    └── example_input.csv       # template showing the exact input format
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. A GPU is recommended for the ESM-2 embedding step but
not required.

## Input file format

One row per residue, covering the entire, unbroken sequence of each
protein. Do not pre-filter by SASA or any other cutoff, filtering is applied
automatically by the tool, after embeddings are generated.

| Column                  | Type    | Notes                                          |
|--------------------------|---------|-------------------------------------------------|
| `sheet`                  | string  | protein/group ID, shared across its residues     |
| `amino acid number`      | integer | 1-indexed, sequential, no gaps                   |
| `amino acid`             | string  | 1-letter (`A`) or 3-letter (`ALA`) code           |
| `total (sasa)`           | float   | solvent-accessible surface area                  |
| `consurf`                | float   | ConSurf conservation score                       |
| `Concavity`              | float   | concavity score (also accepts `concavity score`) |
| `pool rank normalized`   | float   | normalized pool rank (auto-derived from raw `pool rank` if missing, see below) |
| `PPrint Score`           | float   | SVM value from PPrint (also accepts `svm value`) |

See `example/example_input.csv` for a working example.

Column names are matched flexibly (case-insensitive, ignoring `_`/`-`/extra
spacing). If you only have raw `Pool Rank` (not normalized), `predict_rbp.py`
computes it automatically per protein: `(rank - 1) / (max_rank - 1)`.

## Quick start

```bash
python predict_rbp.py \
  --input_csv your_data.csv \
  --artifacts_dir rbp_artifacts \
  --model_path model.joblib \
  --output_csv predictions.csv
```

Output (`predictions.csv`): one row per residue with `sheet`, residue number,
amino acid, SASA, `went_through_model`, `predicted_binding` (0/1), and
`binding_probability`.

## How predict_rbp.py works

1. Validates and fuzzy-matches input columns
2. Derives `pool rank normalized` from raw `pool rank` if not present
3. Reconstructs each protein's full sequence from the input rows
4. Runs ESM-2 (`esm2_t33_650M_UR50D`, layer 33) to get per-residue embeddings
5. Transforms embeddings with the saved PCA
6. Maps RNA-binding propensity from the saved lookup table
7. Assembles the feature vector in the exact order used during training
8. Applies the SASA rule: residues with SASA below the cutoff are buried and
   physically cannot contact RNA, so they are automatically labeled
   non-binder and skip the model entirely, matching how the model was
   trained (it never saw buried residues during training either)
9. Runs the model and applies the validated best-MCC threshold
10. Saves `predictions.csv`

## Visualizing predictions

`color_predictions.py` colors a protein structure in PyMOL directly from a
`predict_rbp.py` output file, predicted binders in green, non-binders in
grey. It auto-detects any CSV in the current folder with a
`predicted_binding` column, so no filename needs to be specified.

```
run color_predictions.py
```
(run from inside PyMOL, in the same folder as your predictions file)

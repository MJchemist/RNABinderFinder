# ===============================================================
# predict_rbp.py
#
# PUBLIC SCRIPT - what other users run.
#
# Given a CSV with the 5 base features (SASA, ConSurf, Concavity,
# pool rank normalized, PPrint Score) plus amino acid, protein
# ID, and residue number columns, this script:
#   1) reconstructs each protein's sequence
#   2) runs ESM2-t33-650M-UR50D to get per-residue embeddings
#   3) TRANSFORMS (does not re-fit) those embeddings with your
#      saved PCA-300
#   4) MAPS (does not recompute) RNA-binding propensity from your
#      saved lookup table
#   5) runs your trained model and outputs predictions
#
# No train/test split here - this is inference only. The user's
# data is not expected to have labels.
#
# REQUIREMENTS:
#   pip install fair-esm torch pandas numpy scikit-learn joblib imbalanced-learn
#
# USAGE:
#   python predict_rbp.py \
#       --input_csv new_proteins.csv \
#       --artifacts_dir rbp_artifacts \
#       --model_path model.joblib \
#       --output_csv predictions.csv
# ===============================================================

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

ESM_LAYER = 33
ESM_EMBED_DIM = 1280

# Selected via evaluate_at_thresholds() -> best_MCC, on the held-out test set
# for the winning config (raw propensity, PCA-300, duplication 1:4).
# sklearn's .predict() default of 0.5 is NOT used here -- it would be far
# too strict given the ~1:16 class imbalance and would understate recall
# relative to your reported/validated test performance.
BEST_MCC_THRESHOLD = 0.20213215

BASE_ATOMS = [
    "pool rank normalized",
    "total (sasa)",
    "consurf",
    "Concavity",
    "PPrint Score",
]

# Accepted alternate names for base features, so column renames don't
# require any change to the trained model itself (the model only sees
# numeric values in a fixed order, never the column names).
FEATURE_ALIASES = {
    "PPrint Score": ["svm value"],
    "Concavity": ["concavity score"],
}


def _norm(s):
    s = str(s).strip().lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s)


def resolve_col(preferred, frame, fallbacks=None):
    fallbacks = fallbacks or []
    wanted = [_norm(preferred)] + [_norm(x) for x in fallbacks]
    norm_map = {_norm(c): c for c in frame.columns}
    for w in wanted:
        if w in norm_map:
            return norm_map[w]
    for k, v in norm_map.items():
        if any(w in k for w in wanted):
            return v
    raise KeyError(
        f"Required column not found. Tried: {[preferred] + fallbacks}. "
        f"Available: {list(frame.columns)}"
    )


def ensure_pool_rank_normalized(df, group_col):
    """
    If 'pool rank normalized' isn't present, derive it from a raw
    'pool rank' column, per protein:
        normalized = (rank - 1) / (max(rank in that protein) - 1)
    Matches the exact formula verified against the original training
    data (best-ranked residue -> 0, worst-ranked -> 1, regardless of
    protein size).
    """
    try:
        return resolve_col("pool rank normalized", df)
    except KeyError:
        pass

    raw_col = resolve_col("pool rank", df, fallbacks=["pool_rank", "poolrank"])
    print(f"'pool rank normalized' not found -- deriving it from "
          f"'{raw_col}' (per-protein: (rank-1)/(max_rank-1))")
    df["pool rank normalized"] = df.groupby(group_col)[raw_col].transform(
        lambda r: (r - 1) / (r.max() - 1)
    )
    return "pool rank normalized"


def aa_to_single(aa):
    aa = str(aa).strip().upper()
    if len(aa) == 1:
        return aa
    if aa in AA3_TO_AA1:
        return AA3_TO_AA1[aa]
    raise ValueError(f"Unrecognized amino acid code: {aa}")


def build_sequences(df, group_col, resnum_col, aa_col):
    sequences, row_order = {}, {}
    for group_id, grp in df.groupby(group_col):
        grp_sorted = grp.sort_values(resnum_col)
        seq = "".join(aa_to_single(a) for a in grp_sorted[aa_col])
        sequences[group_id] = seq
        row_order[group_id] = grp_sorted.index.tolist()
    return sequences, row_order


def run_esm2(sequences):
    import esm
    print("Loading ESM2-t33-650M-UR50D ...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    embeddings = {}
    items = list(sequences.items())
    for i, (name, seq) in enumerate(items):
        print(f"[{i+1}/{len(items)}] Embedding {name} (length {len(seq)})")
        batch = [(str(name), seq)]
        _, _, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            out = model(batch_tokens, repr_layers=[ESM_LAYER], return_contacts=False)
        emb = out["representations"][ESM_LAYER][0, 1:len(seq) + 1]
        embeddings[name] = emb.cpu().numpy()
    return embeddings


def align_embeddings(df, row_order, embeddings):
    X_full = np.zeros((len(df), ESM_EMBED_DIM))
    for group_id, idxs in row_order.items():
        emb = embeddings.get(group_id)
        if emb is None:
            print(f"WARNING: no embedding for {group_id}; filling zeros")
            emb = np.zeros((len(idxs), ESM_EMBED_DIM))
        if len(emb) != len(idxs):
            n = min(len(emb), len(idxs))
            X_full[idxs[:n]] = emb[:n]
            continue
        X_full[idxs] = emb
    return X_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--artifacts_dir", required=True,
                    help="Folder containing pca300.joblib and propensity_raw.json")
    ap.add_argument("--model_path", required=True,
                    help="Path to your trained model.joblib")
    ap.add_argument("--output_csv", default="predictions.csv")
    ap.add_argument("--intermediate_csv", default=None,
                    help="Optional: also save the model-ready feature CSV (for debugging)")
    ap.add_argument("--group_col", default="sheet")
    ap.add_argument("--resnum_col", default="amino acid number")
    ap.add_argument("--aa_col", default="amino acid")
    ap.add_argument("--threshold", type=float, default=BEST_MCC_THRESHOLD,
                    help=f"Probability cutoff for predicted_binding (default: "
                         f"{BEST_MCC_THRESHOLD}, the validated best-MCC threshold)")
    ap.add_argument("--sasa_cutoff", type=float, default=1.0,
                    help="Residues with SASA below this are auto-labeled "
                         "non-binder and skip the model entirely (default: 1.0, "
                         "matching training)")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)

    print(f"Loading input: {args.input_csv}")
    df = pd.read_csv(args.input_csv, low_memory=False)
    print(f"Loaded: {df.shape}")

    group_col = resolve_col(args.group_col, df)
    resnum_col = resolve_col(args.resnum_col, df, fallbacks=["residue number", "resnum"])
    aa_col = resolve_col(args.aa_col, df)
    ensure_pool_rank_normalized(df, group_col)
    base_feature_cols = [resolve_col(f, df, fallbacks=FEATURE_ALIASES.get(f, [])) for f in BASE_ATOMS]
    sasa_col = resolve_col("total (sasa)", df)

    # ── Step 1: sequences + ESM2 ─────────────────────────────────────────
    print("\nReconstructing sequences ...")
    sequences, row_order = build_sequences(df, group_col, resnum_col, aa_col)
    print(f"Reconstructed {len(sequences)} protein sequences")
    embeddings = run_esm2(sequences)
    X_full = align_embeddings(df, row_order, embeddings)

    # ── Step 2: transform with SAVED PCA (no re-fitting) ────────────────
    print("\nLoading saved PCA-300 and transforming (not re-fitting) ...")
    pca = joblib.load(artifacts_dir / "pca300.joblib")
    X_pca = pca.transform(X_full)
    pca_cols = {f"esm2_pca300_{i}": X_pca[:, i] for i in range(X_pca.shape[1])}
    df_pca = pd.DataFrame(pca_cols, index=df.index)

    # ── Step 3: map SAVED propensity (no recomputation) ─────────────────
    print("Loading saved propensity table and mapping ...")
    with open(artifacts_dir / "propensity_raw.json") as f:
        propensity = json.load(f)
    unseen = set(df[aa_col].astype(str).str.upper().unique()) - set(propensity.keys())
    if unseen:
        print(f"WARNING: amino acids not in saved propensity table (defaulting to 0): {unseen}")
    df["rna_propensity_raw"] = df[aa_col].astype(str).str.upper().map(
        lambda a: propensity.get(a, 0.0)
    )

    # ── Step 4: assemble feature matrix in the EXACT column order used in training ──
    numeric_cols = base_feature_cols + ["rna_propensity_raw"]
    X_model = pd.concat([df[numeric_cols].reset_index(drop=True),
                         df_pca.reset_index(drop=True)], axis=1)

    if args.intermediate_csv:
        pd.concat([df.reset_index(drop=True), df_pca.reset_index(drop=True)], axis=1) \
          .to_csv(args.intermediate_csv, index=False)
        print(f"Saved intermediate model-ready CSV: {args.intermediate_csv}")

    # ── Step 5: SASA filter -- buried residues never go through the model ──
    # SASA < cutoff  -> auto non-binder (0), model never sees these
    # SASA >= cutoff -> run through the trained model, exactly like training
    sasa_values = pd.to_numeric(df[sasa_col], errors="coerce")
    exposed_mask = (sasa_values >= args.sasa_cutoff).values
    n_buried = (~exposed_mask).sum()
    print(f"\nSASA cutoff = {args.sasa_cutoff}: "
          f"{exposed_mask.sum()} residues -> model, "
          f"{n_buried} buried residues -> auto non-binder (skip model)")

    y_pred = np.zeros(len(df), dtype=int)
    y_proba = np.zeros(len(df), dtype=float)

    print(f"\nLoading trained model: {args.model_path}")
    model = joblib.load(args.model_path)

    if exposed_mask.any():
        X_exposed = X_model.loc[exposed_mask]
        proba_exposed = model.predict_proba(X_exposed)[:, 1]
        # NOTE: intentionally NOT using model.predict() here -- its default
        # 0.5 threshold is far stricter than the validated best-MCC threshold,
        # and would understate recall given the ~1:16 class imbalance.
        pred_exposed = (proba_exposed >= args.threshold).astype(int)
        y_proba[exposed_mask] = proba_exposed
        y_pred[exposed_mask] = pred_exposed
    print(f"Using classification threshold: {args.threshold} (exposed residues only)")

    out = df[[group_col, resnum_col, aa_col]].copy()
    out[sasa_col] = sasa_values
    out["went_through_model"] = exposed_mask
    out["predicted_binding"] = y_pred
    out["binding_probability"] = y_proba
    out.to_csv(args.output_csv, index=False)
    print(f"\nSaved predictions: {args.output_csv}  ({len(out)} residues)")


if __name__ == "__main__":
    main()

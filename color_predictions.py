# ===============================================================
# color_predictions.py
#
# Colors a protein structure by RNA-binding predictions from
# predict_rbp.py's output.
#
#   - Predicted binder      (predicted_binding == 1)  -> GREEN
#   - Predicted non-binder  (predicted_binding == 0)  -> GREY
#
# Fetches the structure directly from RCSB by PDB ID (the 'sheet'
# value), colors the backbone grey, then recolors binder residues
# green as sticks so they stand out. Saves one .pse session per
# protein found in each predictions file.
#
# NO FILENAME NEEDED: automatically finds every CSV in the current
# folder that has a 'predicted_binding' column and processes all
# of them.
#
# NO PANDAS NEEDED: uses only Python's built-in csv module, so it
# works in PyMOL's own bundled Python environment with no extra
# packages to install.
#
# USAGE:
#   From a regular terminal:      pymol -cq color_predictions.py
#   From inside PyMOL's console:  run color_predictions.py
#
# Optional: pass one or more specific files if you only want those:
#   pymol -cq color_predictions.py -- predictions_holdout.csv
# ===============================================================

from pymol import cmd
import csv
import glob
import sys
import os
from collections import defaultdict

OUTPUT_DIR = "colored_sessions"

BINDER_COLOR = "green"
NONBINDER_COLOR = "grey70"
BACKBONE_COLOR = "grey80"


def find_prediction_csvs():
    """Find every CSV in the current folder with a predicted_binding column."""
    candidates = []
    for path in glob.glob("*.csv"):
        try:
            with open(path, newline="") as f:
                header = next(csv.reader(f))
        except Exception:
            continue
        if "predicted_binding" in header:
            candidates.append(path)
    return candidates


def resolve_field(fieldnames, preferred, fallbacks):
    wanted = [preferred] + fallbacks
    norm_map = {f.strip().lower(): f for f in fieldnames}
    for w in wanted:
        if w.lower() in norm_map:
            return norm_map[w.lower()]
    return None


# If specific file(s) were passed after "--", use only those.
# Otherwise auto-detect every predictions CSV in the folder.
passed_args = [a for a in sys.argv[1:] if a.endswith(".csv")]
csv_files = passed_args if passed_args else find_prediction_csvs()

if not csv_files:
    print("No predictions CSV found (looked for a *.csv file with a "
          "'predicted_binding' column in the current folder).")
    sys.exit(0)

print(f"Found {len(csv_files)} predictions file(s): {csv_files}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for PREDICTIONS_CSV in csv_files:
    print(f"\n=== Processing {PREDICTIONS_CSV} ===")

    with open(PREDICTIONS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        group_col = resolve_field(fieldnames, "sheet", []) or fieldnames[0]
        resnum_col = resolve_field(fieldnames, "amino acid number", ["residue number"])

        # protein_id -> {"binder": [resnums], "nonbinder": [resnums]}
        by_protein = defaultdict(lambda: {"binder": [], "nonbinder": []})
        for row in reader:
            pdb_id = row[group_col]
            resnum = int(float(row[resnum_col]))
            if row["predicted_binding"].strip() in ("1", "1.0"):
                by_protein[pdb_id]["binder"].append(resnum)
            else:
                by_protein[pdb_id]["nonbinder"].append(resnum)

    for pdb_id, groups in by_protein.items():
        binder_resnums = groups["binder"]
        nonbinder_resnums = groups["nonbinder"]

        print(f"\n{pdb_id}: {len(binder_resnums)} predicted binders, "
              f"{len(nonbinder_resnums)} predicted non-binders")

        cmd.reinitialize()
        try:
            cmd.fetch(pdb_id, type="pdb", async_=0)
        except Exception as e:
            print(f"  WARNING: could not fetch {pdb_id}: {e}")
            continue

        obj_name = pdb_id.lower()

        # Remove crystallographic waters and hide any other heteroatoms
        # (ions, ligands, etc.) so only the protein cartoon is visible.
        cmd.remove("solvent")
        cmd.hide("everything", "hetatm")

        cmd.hide("everything", obj_name)
        cmd.show("cartoon", obj_name)
        cmd.color(BACKBONE_COLOR, obj_name)
        cmd.set("cartoon_transparency", 0.3, obj_name)

        if binder_resnums:
            resi_sel = "+".join(str(r) for r in binder_resnums)
            cmd.select("binders", f"{obj_name} and resi {resi_sel}")
            cmd.show("sticks", "binders")
            cmd.color(BINDER_COLOR, "binders")

        if nonbinder_resnums:
            resi_sel = "+".join(str(r) for r in nonbinder_resnums)
            cmd.select("nonbinders", f"{obj_name} and resi {resi_sel}")
            cmd.color(NONBINDER_COLOR, "nonbinders")

        cmd.bg_color("white")
        cmd.orient(obj_name)

        out_path = os.path.join(OUTPUT_DIR, f"{pdb_id}_predicted.pse")
        cmd.save(out_path)
        print(f"  Saved: {out_path}")

print("\nDone. Open each .pse in PyMOL to view.")



"""
Run the full analysis for "The Killing Cascade" (Nix & Adams, Police Quarterly).

Starts from the processed data in data/processed/ and writes every table, figure, and
inline statistic the paper reports to analysis/outputs/.

Usage (from the repository root):
    python analysis/scripts/run_all.py              # everything
    python analysis/scripts/run_all.py --skip-maps  # skip the county maps (need geopandas + internet)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ANALYSIS = SCRIPTS.parent

PIPELINE = [
    # Study 1: Texas
    "study1_texas/01_prepare_data.py",
    "study1_texas/02_run_analysis.py",
    "study1_texas/03_robustness.py",
    "study1_texas/04_geographic_analysis.py",
    # Study 2: California
    "study2_california/01_prepare_data.py",
    "study2_california/02_run_analysis.py",
    "study2_california/03_mediation_analysis.py",
    "study2_california/04_robustness.py",
    "study2_california/05_geographic_analysis.py",
    # Pooled models
    "pooled_analysis.py",
    # Supplementary analyses and validation checks added during review
    "revision/01_nonlinear_officer_count.py",
    "revision/02_bootstrap_mediation.py",
    "revision/03_pooled_interactions.py",
    "revision/04_officer_count_by_race.py",
    "revision/05_race_wound_crosstab.py",
    "revision/06_logistic_invariance.py",
    "revision/07_placebo_falsification.py",
    "revision/08_mechanism_discrimination.py",
    "revision/09_oster_bounds.py",
    "revision/10_predicted_fatality.py",
    "revision/12_sample_flow.py",
    "revision/13_rear_wound_race.py",
    "revision/14_collinearity_diagnostics.py",
    "revision/15_singleton_agency_check.py",
    "revision/17_ca_agency_id_audit.py",
    "revision/18_tx_linkage_sensitivity.py",
    "revision/19_race_heterogeneity_fe.py",
    # Figures
    "create_figures.py",
    "create_map_panels.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-maps", action="store_true")
    args = parser.parse_args()

    total = time.time()
    for i, rel in enumerate(PIPELINE, 1):
        if args.skip_maps and rel == "create_map_panels.py":
            continue
        cmd = [sys.executable, str(SCRIPTS / rel)]
        print(f"[{i:2d}/{len(PIPELINE)}] {rel}", flush=True)
        start = time.time()
        result = subprocess.run(cmd, cwd=ANALYSIS.parent, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout[-3000:])
            print(result.stderr[-3000:])
            sys.exit(f"FAILED: {rel} (exit {result.returncode})")
        print(f"        ok ({time.time() - start:.0f}s)", flush=True)

    print(f"Done in {(time.time() - total) / 60:.1f} min. Outputs are in analysis/outputs/.")


if __name__ == "__main__":
    main()

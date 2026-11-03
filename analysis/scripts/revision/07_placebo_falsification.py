"""
REVISION: Balance checks for officer count.

Regresses civilian and incident characteristics on officer count to show how incidents with
more officers differ from incidents with fewer. Revised 2026-10-01 for PQ Round 2:
  - Renamed from "placebo / falsification" to "balance checks" (Reviewer 2): balance on a few
    demographics says little about balance on incident severity, and these checks do not
    resolve selection.
  - Adds severity markers as outcomes (armed; civilian fired at / shot an officer; firearm
    recorded), which show imbalance on severity directly.
  - Primary specification throughout (year FE, agency-clustered SEs, regression samples), plus
    an agency-FE version in both states. The direction of each coefficient is stated in the
    output.

Outputs:
  outputs/revision/placebo_test_results.csv   (file name kept for downstream references)
"""

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from lpm_utils import fit_lpm  # noqa: E402

MERGED_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = MERGED_ROOT / 'outputs' / 'revision'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tx = pd.read_csv(MERGED_ROOT / 'outputs' / 'study1' / 'texas_regression_data.csv')
ca = pd.read_csv(MERGED_ROOT / 'outputs' / 'study2' / 'california_regression_data.csv')

DEMOGRAPHIC = ['age_10yr', 'female', 'race_black', 'race_hispanic']
specs = [
    ('Texas', tx, 'Officer_Count', DEMOGRAPHIC + ['armed', 'officer_shot_by_civ', 'weapon_firearm']),
    ('California', ca, 'num_involved_officers',
     DEMOGRAPHIC + ['armed_assault', 'civ_fired_at_officer', 'weapon_firearm']),
]

rows = []
for state, df, oc, outcomes in specs:
    for agency_fe in (False, True):
        for outcome in outcomes:
            res = fit_lpm(df, outcome, [oc], agency_fe=agency_fe)
            scale = 1 if outcome == 'age_10yr' else 100
            ci = res.conf_int().loc[oc] * scale
            coef = res.params[oc] * scale
            rows.append({
                'state': state + (' (agency FE)' if agency_fe else ''),
                'outcome': outcome,
                'type': 'demographic' if outcome in DEMOGRAPHIC else 'severity marker',
                'officer_coef': coef, 'ci_lower': ci[0], 'ci_upper': ci[1],
                'p_value': res.pvalues[oc],
                'unit': 'decades per +1 officer' if outcome == 'age_10yr' else 'pp per +1 officer',
                'direction': ('more likely with more officers' if coef > 0
                              else 'less likely with more officers'),
                'n': int(res.nobs), 'n_clusters': res.n_clusters,
            })

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_DIR / 'placebo_test_results.csv', index=False)
pd.set_option('display.width', 200)
print(results[['state', 'outcome', 'officer_coef', 'ci_lower', 'ci_upper', 'p_value', 'direction']]
      .round(3).to_string(index=False))

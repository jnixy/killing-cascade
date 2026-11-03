"""
REVISION: Agency fixed-effects models with and without single-case agencies.

With agency fixed effects, an agency contributing one case is absorbed by its own dummy and
adds nothing to the within-agency estimate, but it still counts as a cluster (Correia 2015
recommends dropping such singletons). This refits each agency-FE model (TX M5; CA M5-M7)
with and without singleton agencies. Coefficients are identical by construction; the
comparison is of the agency-clustered standard errors. Cited in a Methods footnote.

Output:
  outputs/revision/singleton_agency_check.csv
    state, model, sample, n, n_agencies, coef_pp, se_pp, p
"""

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from lpm_utils import fit_lpm  # noqa: E402

MERGED_ROOT = SCRIPT_DIR.parent.parent
OUT = MERGED_ROOT / 'outputs'
OUTPUT_DIR = OUT / 'revision'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# M4 covariates (agency FE added below), as in study1_texas/02 and study2_california/02
TX_M4 = ['Officer_Count', 'age_10yr', 'female', 'race_black', 'race_hispanic', 'armed',
         'officer_shot_by_civ', 'weapon_firearm', 'weapon_knife', 'weapon_vehicle', 'weapon_other',
         'call_traffic', 'call_warrant', 'call_other', 'multi_civilian']
CA_M4 = ['num_involved_officers', 'age_10yr', 'female', 'race_black', 'race_hispanic',
         'armed_assault', 'civ_fired_at_officer', 'civ_threat_firearm', 'officer_injured', 'civ_vehicle_against_officer',
         'weapon_firearm', 'weapon_replica', 'weapon_knife', 'weapon_other',
         'contact_stop', 'contact_planned', 'contact_other', 'multi_civilian']
WOUND = ['wound_head_neck', 'wound_chest', 'wound_abdomen', 'wound_back']

SPECS = [
    ('Texas', OUT / 'study1' / 'texas_regression_data.csv', [('M5', TX_M4)]),
    ('California', OUT / 'study2' / 'california_regression_data.csv',
     [('M5', CA_M4), ('M6', CA_M4 + WOUND), ('M7', CA_M4 + WOUND + ['wound_count'])]),
]

rows = []
for state, path, models in SPECS:
    d = pd.read_csv(path, low_memory=False)
    widest = max((x for _, x in models), key=len)
    d = d.dropna(subset=widest + ['fatal', 'year', 'agency'])
    multi = d.groupby('agency')['agency'].transform('size') > 1
    for model, xvars in models:
        oc = xvars[0]
        for sample, dd in [('all agencies', d), ('singletons dropped', d[multi])]:
            m = fit_lpm(dd, 'fatal', xvars, agency_fe=True)
            rows.append({'state': state, 'model': model, 'sample': sample, 'n': int(m.nobs),
                         'n_agencies': int(m.n_clusters), 'coef_pp': m.params[oc] * 100,
                         'se_pp': m.bse[oc] * 100, 'p': m.pvalues[oc]})

res = pd.DataFrame(rows)
res.to_csv(OUTPUT_DIR / 'singleton_agency_check.csv', index=False)
print(res.round(3).to_string(index=False))

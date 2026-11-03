"""
REVISION: Collinearity diagnostics for the M4 covariates (both states).

The incident-circumstance controls overlap conceptually (weapon status, firearm recorded,
civilian shot / fired at an officer), so this reports variance inflation factors for every
M4 covariate (computed with the incident-year dummies included) and pairwise correlations
among officer count and the weapon/threat controls. Cited in a Measures footnote.

Outputs:
  outputs/revision/collinearity_vif.csv           state, variable, vif
  outputs/revision/collinearity_correlations.csv  state, var1, var2, r
"""

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from lpm_utils import fe_dummies  # noqa: E402

MERGED_ROOT = SCRIPT_DIR.parent.parent
OUT = MERGED_ROOT / 'outputs'
OUTPUT_DIR = OUT / 'revision'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# M4 covariates, as in study1_texas/02 and study2_california/02
SPECS = {
    'Texas': {
        'data': OUT / 'study1' / 'texas_regression_data.csv',
        'm4': ['Officer_Count', 'age_10yr', 'female', 'race_black', 'race_hispanic', 'armed',
               'officer_shot_by_civ', 'weapon_firearm', 'weapon_knife', 'weapon_vehicle',
               'weapon_other', 'call_traffic', 'call_warrant', 'call_other', 'multi_civilian'],
        'focus': ['Officer_Count', 'armed', 'weapon_firearm', 'officer_shot_by_civ'],
    },
    'California': {
        'data': OUT / 'study2' / 'california_regression_data.csv',
        'm4': ['num_involved_officers', 'age_10yr', 'female', 'race_black', 'race_hispanic',
               'armed_assault', 'civ_fired_at_officer', 'civ_threat_firearm', 'officer_injured', 'civ_vehicle_against_officer',
               'weapon_firearm', 'weapon_replica', 'weapon_knife', 'weapon_other',
               'contact_stop', 'contact_planned', 'contact_other', 'multi_civilian'],
        'focus': ['num_involved_officers', 'armed_assault', 'weapon_firearm',
                  'civ_fired_at_officer', 'civ_threat_firearm'],
    },
}

vif_rows, cor_rows = [], []
for state, spec in SPECS.items():
    d = pd.read_csv(spec['data'], low_memory=False)
    d = d[spec['m4'] + ['year']].dropna()
    X = sm.add_constant(pd.concat([d[spec['m4']].astype(float), fe_dummies(d, 'year', 'yr')], axis=1))
    for i, col in enumerate(X.columns):
        if col in spec['m4']:
            vif_rows.append({'state': state, 'variable': col, 'vif': variance_inflation_factor(X.values, i)})
    for a, b in combinations(spec['focus'], 2):
        cor_rows.append({'state': state, 'var1': a, 'var2': b, 'r': d[a].astype(float).corr(d[b].astype(float))})

vif = pd.DataFrame(vif_rows)
cor = pd.DataFrame(cor_rows)
vif.to_csv(OUTPUT_DIR / 'collinearity_vif.csv', index=False)
cor.to_csv(OUTPUT_DIR / 'collinearity_correlations.csv', index=False)
for state in SPECS:
    v = vif[vif['state'] == state].sort_values('vif', ascending=False).iloc[0]
    c = cor[cor['state'] == state].assign(a=lambda x: x['r'].abs()).sort_values('a', ascending=False).iloc[0]
    print(f"{state}: max VIF {v['vif']:.2f} ({v['variable']}); "
          f"max |r| {c['r']:.2f} ({c['var1']} x {c['var2']})")

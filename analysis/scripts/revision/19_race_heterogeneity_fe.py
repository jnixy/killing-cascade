"""
REVISION (PQ Round 3): Race x officer-count heterogeneity under the within-agency adjustment.

R2 asked that the Texas race x officer-count interaction and the race-stratified officer-count
slopes be reported under the preferred adjustment (M5: M4 covariates + year FE + agency FE) or be
identified as exploratory. 02_run_analysis.py estimates the Texas F-test with M4 covariates and
year FE only, and the stratified slopes with age, gender, and armed status. This script re-estimates
both, in both states, with the M5 adjustment, and keeps the year-FE-only (M4) version alongside
for comparison. Agency-clustered SEs throughout (lpm_utils.fit_lpm).

Output: outputs/revision/race_heterogeneity_fe.csv
  one row per state x adjustment x estimate (F-test, or slope for White/Black/Hispanic)
"""

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from lpm_utils import fit_lpm, coef_row  # noqa: E402

MERGED_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = MERGED_ROOT / 'outputs' / 'revision'

# M4 covariate sets, as in each state's 02_run_analysis.py
STATES = {
    'Texas': {
        'data': MERGED_ROOT / 'outputs' / 'study1' / 'texas_regression_data.csv',
        'officer': 'Officer_Count',
        'm4': ['age_10yr', 'female', 'race_black', 'race_hispanic', 'armed',
               'officer_shot_by_civ', 'weapon_firearm', 'weapon_knife', 'weapon_vehicle',
               'weapon_other', 'call_traffic', 'call_warrant', 'call_other', 'multi_civilian'],
    },
    'California': {
        'data': MERGED_ROOT / 'outputs' / 'study2' / 'california_regression_data.csv',
        'officer': 'num_involved_officers',
        'm4': ['age_10yr', 'female', 'race_black', 'race_hispanic', 'armed_assault',
               'civ_fired_at_officer', 'civ_threat_firearm', 'officer_injured',
               'civ_vehicle_against_officer', 'weapon_firearm', 'weapon_replica', 'weapon_knife',
               'weapon_other', 'contact_stop', 'contact_planned', 'contact_other', 'multi_civilian'],
    },
}
ADJUSTMENTS = {'M4 (year FE)': False, 'M5 (year + agency FE)': True}

rows = []
for state, cfg in STATES.items():
    df = pd.read_csv(cfg['data'])
    off = cfg['officer']
    df['off_x_black'] = df[off] * df['race_black']
    df['off_x_hispanic'] = df[off] * df['race_hispanic']
    for adj, agency_fe in ADJUSTMENTS.items():
        res = fit_lpm(df, 'fatal', [off] + cfg['m4'] + ['off_x_black', 'off_x_hispanic'], agency_fe=agency_fe)
        f = res.f_test('off_x_black = 0, off_x_hispanic = 0')
        rows.append({'state': state, 'adjustment': adj, 'estimate': 'Race x officer count F-test',
                     'n': int(res.nobs), 'stat': float(f.fvalue), 'df': 2, 'p': float(f.pvalue)})
        # Stratified slopes: same covariates minus the race indicators
        within = [v for v in cfg['m4'] if v not in ('race_black', 'race_hispanic')]
        for race in ('White', 'Black', 'Hispanic'):
            sub = df[df['race_std'] == race]
            r = fit_lpm(sub, 'fatal', [off] + within, agency_fe=agency_fe)
            rows.append({'state': state, 'adjustment': adj, 'estimate': f'Slope: {race}',
                         'n': int(r.nobs), 'n_clusters': r.n_clusters, **coef_row(r, off)})

out = pd.DataFrame(rows)
out.to_csv(OUTPUT_DIR / 'race_heterogeneity_fe.csv', index=False)
print(out.round(3).to_string(index=False))

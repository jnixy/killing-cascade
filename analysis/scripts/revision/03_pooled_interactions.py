"""
REVISION: Pooled interaction tests.

1. State x officer count interaction (does the slope differ between TX and CA?)
2. Do the TX and CA Black coefficients differ? (Wald z from separate state models)
3. Change in the CA Black coefficient across the model sequence

Revised 2026-10-01 for PQ Round 2: primary specification (state-by-year FE, agency-clustered
SEs) on the two state regression samples, matching pooled_analysis.py. Note the exposures
differ by state (TX: officers who discharged; CA: officers involved), so the interaction tests
whether the two measures relate to fatality similarly, not whether one quantity is homogeneous.

Outputs (names kept for downstream references):
  outputs/revision/state_officer_interaction.csv
  outputs/revision/state_black_interaction.csv
  outputs/revision/m2_m3_black_gap_change.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from lpm_utils import fit_lpm  # noqa: E402

MERGED_ROOT = SCRIPT_DIR.parent.parent
OUT = MERGED_ROOT / 'outputs'
OUTPUT_DIR = OUT / 'revision'

tx = pd.read_csv(OUT / 'study1' / 'texas_regression_data.csv')
ca = pd.read_csv(OUT / 'study2' / 'california_regression_data.csv')
tx['state'], tx['officer_count'] = 'Texas', tx['Officer_Count']
ca['state'], ca['officer_count'], ca['armed'] = 'California', ca['num_involved_officers'], ca['armed_assault']
cols = ['state', 'fatal', 'officer_count', 'age_10yr', 'female', 'race_black', 'race_hispanic',
        'armed', 'year', 'agency']
pooled = pd.concat([tx[cols], ca[cols]], ignore_index=True)
pooled['state_ca'] = (pooled['state'] == 'California').astype(int)
pooled['year'] = pooled['state'].str[:2] + '_' + pooled['year'].astype(int).astype(str)
pooled['agency'] = pooled['state'].str[:2] + '_' + pooled['agency'].astype(str)
base = ['officer_count', 'age_10yr', 'female', 'race_black', 'race_hispanic', 'armed']

# --- 1. State x officer count (state main effect absorbed by state-by-year FE) ---
pm3 = fit_lpm(pooled, 'fatal', base)
pooled['state_x_officer'] = pooled['state_ca'] * pooled['officer_count']
pm3_int = fit_lpm(pooled, 'fatal', base + ['state_x_officer'])
ci = pm3_int.conf_int().loc['state_x_officer'] * 100
state_officer = {
    'interaction_coef_pp': pm3_int.params['state_x_officer'] * 100,
    'interaction_se_pp': pm3_int.bse['state_x_officer'] * 100,
    'interaction_ci_lower': ci[0], 'interaction_ci_upper': ci[1],
    'interaction_p': pm3_int.pvalues['state_x_officer'],
    'tx_slope_pp': pm3_int.params['officer_count'] * 100,
    'ca_slope_pp': (pm3_int.params['officer_count'] + pm3_int.params['state_x_officer']) * 100,
    'pooled_slope_pp': pm3.params['officer_count'] * 100,
    'pooling_justified': pm3_int.pvalues['state_x_officer'] > 0.05,
    'N': int(pm3.nobs),
}
pd.DataFrame([state_officer]).to_csv(OUTPUT_DIR / 'state_officer_interaction.csv', index=False)
print(f"State x officer: {state_officer['interaction_coef_pp']:+.2f} pp "
      f"[{ci[0]:.2f}, {ci[1]:.2f}], p = {state_officer['interaction_p']:.3f}")

# --- 2. TX vs CA Black coefficient (M3 in each state) ---
tx_m = fit_lpm(tx, 'fatal', ['Officer_Count', 'age_10yr', 'female', 'race_black', 'race_hispanic', 'armed'])
ca_m = fit_lpm(ca, 'fatal', ['num_involved_officers', 'age_10yr', 'female', 'race_black',
                             'race_hispanic', 'armed_assault'])
b_tx, se_tx = tx_m.params['race_black'], tx_m.bse['race_black']
b_ca, se_ca = ca_m.params['race_black'], ca_m.bse['race_black']
z = (b_tx - b_ca) / np.sqrt(se_tx ** 2 + se_ca ** 2)
state_black = {'tx_black_pp': b_tx * 100, 'tx_black_se': se_tx * 100,
               'ca_black_pp': b_ca * 100, 'ca_black_se': se_ca * 100,
               'difference_pp': (b_tx - b_ca) * 100, 'wald_z': z,
               'wald_p': 2 * stats.norm.sf(abs(z))}
pd.DataFrame([state_black]).to_csv(OUTPUT_DIR / 'state_black_interaction.csv', index=False)
print(f"Black: TX {b_tx*100:.2f}, CA {b_ca*100:.2f}, z = {z:.2f}, p = {state_black['wald_p']:.3f}")

# --- 3. CA Black coefficient across the model sequence ---
ca_models = pd.read_csv(OUT / 'study2' / 'tables' / 'regression_models.csv')
seq = ca_models[ca_models['Model'].str.match(r'M[1-7]:')][['Model', 'Black_Coef', 'Black_p']].dropna()
seq.to_csv(OUTPUT_DIR / 'm2_m3_black_gap_change.csv', index=False)
print(seq.round(3).to_string(index=False))

"""
REVISION (PQ Round 2): Sensitivity of the officer-count association to unobserved confounding.

Responds to PQ Reviewer 2's central concern (comment 1): officer count may mark incident
severity that also drives fatality. Following Oster (2019), compute
  - delta: how strongly unobservables would have to be related to officer count, relative to
    the observed controls, for the true coefficient to be zero, and
  - beta*: the bias-adjusted coefficient assuming delta = 1,
with R_max = 1.3 x R-squared of the controlled model (Oster's recommended bound). Because
R-squared is low in these models (0.07-0.08), that rule caps unexplained variance tightly, so
results are also reported for R_max = 0.3 and 0.5 (the latter well above the 0.26 that CA
reaches only after adding post-exposure wound variables).

Uncontrolled model: officer count + year FE. Controlled model: M4 (demographics, armed,
incident circumstances, year FE). CA wound variables are excluded: they are post-exposure,
not confounders. Bias formula uses the closed-form approximation in Oster (2019, Prop. 2).

Output: outputs/revision/oster_bounds.csv
"""

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from lpm_utils import fit_lpm  # noqa: E402

MERGED_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = MERGED_ROOT / 'outputs' / 'revision'
R_MAX_MULT = 1.3

TX_COVS = ['age_10yr', 'female', 'race_black', 'race_hispanic', 'armed',
           'officer_shot_by_civ', 'weapon_firearm', 'weapon_knife', 'weapon_vehicle',
           'weapon_other', 'call_traffic', 'call_warrant', 'call_other', 'multi_civilian']
CA_COVS = ['age_10yr', 'female', 'race_black', 'race_hispanic', 'armed_assault',
           'civ_fired_at_officer', 'civ_threat_firearm', 'officer_injured', 'civ_vehicle_against_officer',
           'weapon_firearm', 'weapon_replica', 'weapon_knife', 'weapon_other',
           'contact_stop', 'contact_planned', 'contact_other', 'multi_civilian']


def oster(df, x, covs, state, rmax_rule):
    short = fit_lpm(df, 'fatal', [x])
    full = fit_lpm(df, 'fatal', [x] + covs)
    b0, r0 = short.params[x], short.rsquared
    b1, r1 = full.params[x], full.rsquared
    rmax = min(1.0, R_MAX_MULT * r1) if rmax_rule == '1.3 x R2' else float(rmax_rule)
    # Oster (2019) approximations: beta* (delta = 1) and delta for beta = 0
    beta_star = b1 - (b0 - b1) * (rmax - r1) / (r1 - r0)
    delta_zero = (b1 * (r1 - r0)) / ((b0 - b1) * (rmax - r1)) if b0 != b1 else float('inf')
    return {'state': state, 'rmax_rule': rmax_rule, 'beta_uncontrolled': b0 * 100, 'r2_uncontrolled': r0,
            'beta_controlled': b1 * 100, 'r2_controlled': r1, 'r_max': rmax,
            'beta_star_delta1': beta_star * 100, 'delta_for_zero': delta_zero}


tx = pd.read_csv(MERGED_ROOT / 'outputs' / 'study1' / 'texas_regression_data.csv')
ca = pd.read_csv(MERGED_ROOT / 'outputs' / 'study2' / 'california_regression_data.csv')
rows = [oster(d, x, c, st, rule)
        for d, x, c, st in [(tx, 'Officer_Count', TX_COVS, 'Texas'),
                            (ca, 'num_involved_officers', CA_COVS, 'California')]
        for rule in ['1.3 x R2', '0.3', '0.5']]
out = pd.DataFrame(rows)
out.to_csv(OUTPUT_DIR / 'oster_bounds.csv', index=False)
pd.set_option('display.width', 200)
print(out.round(3).to_string(index=False))

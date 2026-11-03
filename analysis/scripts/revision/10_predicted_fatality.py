"""
REVISION (PQ Round 2): Adjusted predicted fatality by officer count, and model-based
differences in expected deaths.

Responds to PQ Reviewer 2 ("a single principal table could display the adjusted predicted
fatalities for one, two, three, and four or more officers") and replaces the Round 1
counterfactual simulations, which applied raw cell case-fatality rates and carried no
uncertainty.

Model: LPM with officer-count category dummies (2, 3, 4+; reference = 1), the full M4
covariate set for each state (demographics, armed, incident circumstances), incident-year
fixed effects, and agency-clustered SEs.

Predicted fatality at category k is the average prediction with every case set to k (marginal
standardization). Expected-death differences are in-sample sums of the category coefficients:
  - "No multi-officer premium": every 2+ case at the one-officer prediction.
  - "One fewer officer": every k-officer case at the (k-1) prediction (4+ moved to 3).
Both are linear in the coefficients, so their 95% CIs follow exactly from the cluster-robust
variance matrix. They are descriptive differences under the stated model, not estimates of
preventable deaths.

Outputs:
  outputs/revision/predicted_fatality_by_officers.csv
  outputs/revision/expected_death_differences.csv
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
OUTPUT_DIR = MERGED_ROOT / 'outputs' / 'revision'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATS = ['off_2', 'off_3', 'off_4plus']
LABELS = {'off_1': '1 officer', 'off_2': '2 officers', 'off_3': '3 officers', 'off_4plus': '4+ officers'}
Z = stats.norm.ppf(0.975)

TX_COVS = ['age_10yr', 'female', 'race_black', 'race_hispanic', 'armed',
           'officer_shot_by_civ', 'weapon_firearm', 'weapon_knife', 'weapon_vehicle',
           'weapon_other', 'call_traffic', 'call_warrant', 'call_other', 'multi_civilian']
CA_COVS = ['age_10yr', 'female', 'race_black', 'race_hispanic', 'armed_assault',
           'civ_fired_at_officer', 'civ_threat_firearm', 'officer_injured', 'civ_vehicle_against_officer',
           'weapon_firearm', 'weapon_replica', 'weapon_knife', 'weapon_other',
           'contact_stop', 'contact_planned', 'contact_other', 'multi_civilian']


def linear_combo(res, weights):
    """Estimate, SE, and 95% CI of sum(weights[name] * beta[name]) using the model's vcov."""
    names = list(res.params.index)
    c = np.array([weights.get(n, 0.0) for n in names])
    est = float(c @ res.params.values)
    se = float(np.sqrt(c @ res.cov_params().values @ c))
    return est, se, est - Z * se, est + Z * se


def analyze(df, officer_col, covs, state):
    df = df.copy()
    k = df[officer_col]
    df['off_2'] = (k == 2).astype(int)
    df['off_3'] = (k == 3).astype(int)
    df['off_4plus'] = (k >= 4).astype(int)
    res = fit_lpm(df, 'fatal', CATS + covs)
    X = res.model.exog
    names = list(res.params.index)
    xbar = dict(zip(names, X.mean(axis=0)))

    # Marginal standardization: average covariate profile, category dummies set to k
    pred_rows = []
    for cat in ['off_1'] + CATS:
        w = {n: v for n, v in xbar.items() if n not in CATS}
        if cat != 'off_1':
            w[cat] = 1.0
        est, se, lo, hi = linear_combo(res, w)
        obs = df.loc[df['off_' + cat.split('_')[1]] == 1, 'fatal'] if cat != 'off_1' \
            else df.loc[k == 1, 'fatal']
        pred_rows.append({'state': state, 'category': LABELS[cat], 'n': len(obs),
                          'observed_cfr': obs.mean() * 100, 'predicted_pct': est * 100,
                          'ci_lower': lo * 100, 'ci_upper': hi * 100})
    for cat in CATS:  # contrasts vs one officer
        est, se, lo, hi = linear_combo(res, {cat: 1.0})
        pred_rows.append({'state': state, 'category': f'{LABELS[cat]} minus 1 officer',
                          'n': np.nan, 'observed_cfr': np.nan, 'predicted_pct': est * 100,
                          'ci_lower': lo * 100, 'ci_upper': hi * 100, 'p': 2 * stats.norm.sf(abs(est / se))})

    n2, n3, n4 = (int(df[c].sum()) for c in CATS)
    # Every 2+ case at the one-officer prediction: sum of its category coefficient
    no_premium = {'off_2': n2, 'off_3': n3, 'off_4plus': n4}
    # Each case one officer lower: 2->1 (b2), 3->2 (b3-b2), 4+->3 (b4-b3)
    one_fewer = {'off_2': n2 - n3, 'off_3': n3 - n4, 'off_4plus': n4}
    death_rows = []
    for scen, w in [('No multi-officer premium', no_premium), ('One fewer officer', one_fewer)]:
        est, se, lo, hi = linear_combo(res, w)
        death_rows.append({'state': state, 'scenario': scen, 'n_cases': len(df),
                           'observed_deaths': int(df['fatal'].sum()),
                           'expected_difference': est, 'ci_lower': lo, 'ci_upper': hi,
                           'pct_of_deaths': est / df['fatal'].sum() * 100})
    return pred_rows, death_rows, res


tx = pd.read_csv(MERGED_ROOT / 'outputs' / 'study1' / 'texas_regression_data.csv')
ca = pd.read_csv(MERGED_ROOT / 'outputs' / 'study2' / 'california_regression_data.csv')

pred_all, death_all = [], []
for df, col, covs, state in [(tx, 'Officer_Count', TX_COVS, 'Texas'),
                             (ca, 'num_involved_officers', CA_COVS, 'California')]:
    preds, deaths, _ = analyze(df, col, covs, state)
    pred_all += preds
    death_all += deaths

# Combined total: states are independent samples, so variances add
death_df = pd.DataFrame(death_all)
for scen in death_df['scenario'].unique():
    sub = death_df[death_df['scenario'] == scen]
    se = np.sqrt((((sub['ci_upper'] - sub['ci_lower']) / (2 * Z)) ** 2).sum())
    est = sub['expected_difference'].sum()
    death_all.append({'state': 'Combined', 'scenario': scen, 'n_cases': sub['n_cases'].sum(),
                      'observed_deaths': sub['observed_deaths'].sum(),
                      'expected_difference': est, 'ci_lower': est - Z * se, 'ci_upper': est + Z * se,
                      'pct_of_deaths': est / sub['observed_deaths'].sum() * 100})

pred_df = pd.DataFrame(pred_all)
death_df = pd.DataFrame(death_all)
pred_df.to_csv(OUTPUT_DIR / 'predicted_fatality_by_officers.csv', index=False)
death_df.to_csv(OUTPUT_DIR / 'expected_death_differences.csv', index=False)

pd.set_option('display.width', 160)
print(pred_df.round(2).to_string(index=False))
print(death_df.round(1).to_string(index=False))

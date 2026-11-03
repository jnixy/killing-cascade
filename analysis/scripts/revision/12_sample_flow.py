"""
REVISION (PQ Round 2): Sample flow and missing-data comparison.

Responds to PQ Reviewer 2 (comment 6) and Reviewer 1: reconciles every analytic N and
compares cases excluded from the regression samples with included cases on fatality and
officer count, to assess whether complete-case analysis could bias the estimates.

Outputs:
  outputs/revision/sample_flow.csv
  outputs/revision/excluded_vs_included.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
MERGED_ROOT = SCRIPT_DIR.parent.parent
OUT = MERGED_ROOT / 'outputs'
OUTPUT_DIR = OUT / 'revision'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RACES = ['White', 'Black', 'Hispanic']
flow = []

# --- Texas ---
tx = pd.read_csv(OUT / 'study1' / 'texas_analysis_ready.csv')
steps = [('All OAG-reported civilians shot (TJI)', pd.Series(True, index=tx.index))]
m = tx['race_std'].isin(RACES)
steps.append(('White, Black, or Hispanic', m))
m = m & tx['age_numeric'].notna()
steps.append(('Age recorded', m))
m = m & tx['Victim_Gender'].astype(str).str.upper().isin(['MALE', 'FEMALE'])
steps.append(('Gender recorded (regression sample)', m))
for label, mask in steps:
    flow.append({'state': 'Texas', 'step': label, 'n': int(mask.sum()),
                 'fatal_pct': tx.loc[mask, 'fatal'].mean() * 100})
tx['included'] = m

# --- California ---
ca = pd.read_csv(OUT / 'study2' / 'california_analysis_ready.csv')
steps = [('All civilians hit by police gunfire (URSUS)', pd.Series(True, index=ca.index))]
m = ca['num_involved_officers'].notna()
steps.append(('Officer count recorded', m))
m = m & ca['race_std'].isin(RACES)
steps.append(('White, Black, or Hispanic', m))
m = m & ca['wound_location_std'].isin(['Head/Neck', 'Chest', 'Abdomen', 'Back', 'Extremities'])
steps.append(('Wound region recorded', m))
m = m & ca['age_numeric'].notna()
steps.append(('Age recorded', m))
m = m & ca['gender_std'].isin(['Male', 'Female'])
steps.append(('Gender recorded (regression sample)', m))
for label, mask in steps:
    flow.append({'state': 'California', 'step': label, 'n': int(mask.sum()),
                 'fatal_pct': ca.loc[mask, 'fatal'].mean() * 100})
ca['included'] = m

flow_df = pd.DataFrame(flow)
reg_tx = len(pd.read_csv(OUT / 'study1' / 'texas_regression_data.csv'))
reg_ca = len(pd.read_csv(OUT / 'study2' / 'california_regression_data.csv'))
assert flow_df.query("state == 'Texas'")['n'].iloc[-1] == reg_tx, 'TX flow does not reconcile'
assert flow_df.query("state == 'California'")['n'].iloc[-1] == reg_ca, 'CA flow does not reconcile'
flow_df = pd.concat([flow_df, pd.DataFrame([{'state': 'Pooled', 'step': 'TX + CA regression samples',
                                              'n': reg_tx + reg_ca, 'fatal_pct': np.nan}])])
flow_df.to_csv(OUTPUT_DIR / 'sample_flow.csv', index=False)

# --- Excluded vs included ---
comp = []
for state, d, oc in [('Texas', tx, 'Officer_Count'), ('California', ca, 'num_involved_officers')]:
    inc, exc = d[d['included']], d[~d['included']]
    t = stats.ttest_ind(inc['fatal'], exc['fatal'], equal_var=False)
    exc_oc = exc[oc].dropna()
    t2 = stats.ttest_ind(inc[oc], exc_oc, equal_var=False)
    comp.append({'state': state, 'n_included': len(inc), 'n_excluded': len(exc),
                 'fatal_included': inc['fatal'].mean() * 100, 'fatal_excluded': exc['fatal'].mean() * 100,
                 'fatal_diff_p': t.pvalue,
                 'officers_included': inc[oc].mean(), 'officers_excluded': exc_oc.mean(),
                 'officers_diff_p': t2.pvalue,
                 'pct_multi_included': (inc[oc] >= 2).mean() * 100,
                 'pct_multi_excluded': (exc_oc >= 2).mean() * 100})
comp_df = pd.DataFrame(comp)
comp_df.to_csv(OUTPUT_DIR / 'excluded_vs_included.csv', index=False)

pd.set_option('display.width', 200)
print(flow_df.round(1).to_string(index=False))
print(comp_df.round(3).to_string(index=False))

# --- IPW sensitivity for CA missing-data exclusions ---
# Among in-scope cases (White/Black/Hispanic, officer count recorded), 48 are excluded for a
# missing wound region, age, or gender. Weight included cases by the inverse probability of
# being complete, modeled on fatality, officer count, and race, and re-estimate M4.
import sys  # noqa: E402
import statsmodels.api as sm  # noqa: E402
sys.path.insert(0, str(SCRIPT_DIR.parent))
from lpm_utils import fe_dummies  # noqa: E402

scope = ca[ca['num_involved_officers'].notna() & ca['race_std'].isin(RACES)].copy()
Xp = sm.add_constant(pd.concat([
    scope[['fatal', 'num_involved_officers']].astype(float),
    (scope['race_std'] == 'Black').astype(float).rename('black'),
    (scope['race_std'] == 'Hispanic').astype(float).rename('hispanic')], axis=1))
ps = sm.Logit(scope['included'].astype(int), Xp).fit(disp=0).predict(Xp)
scope['ipw'] = 1 / ps

reg = pd.read_csv(OUT / 'study2' / 'california_regression_data.csv')
reg = reg.merge(scope.loc[scope['included'], ['unified_incident_id', 'race_std', 'age_numeric', 'ipw']]
                .drop_duplicates(['unified_incident_id', 'race_std', 'age_numeric']),
                on=['unified_incident_id', 'race_std', 'age_numeric'], how='left')
reg['ipw'] = reg['ipw'].fillna(1.0)
m4 = ['num_involved_officers', 'age_10yr', 'female', 'race_black', 'race_hispanic', 'armed_assault',
      'civ_fired_at_officer', 'civ_threat_firearm', 'officer_injured', 'civ_vehicle_against_officer', 'weapon_firearm',
      'weapon_replica', 'weapon_knife', 'weapon_other', 'contact_stop', 'contact_planned',
      'contact_other', 'multi_civilian']
X = sm.add_constant(pd.concat([reg[m4].astype(float), fe_dummies(reg, 'year', 'yr')], axis=1))
groups = pd.factorize(reg['agency'])[0]
ipw_fit = sm.WLS(reg['fatal'], X, weights=reg['ipw']).fit(cov_type='cluster', cov_kwds={'groups': groups})
ci = ipw_fit.conf_int().loc['num_involved_officers'] * 100
ipw_row = pd.DataFrame([{'model': 'CA M4, IPW for missing wound/age/gender',
                         'n': int(ipw_fit.nobs), 'n_missing_in_scope': int((~scope['included']).sum()),
                         'officer_coef': ipw_fit.params['num_involved_officers'] * 100,
                         'ci_lower': ci[0], 'ci_upper': ci[1],
                         'black_coef': ipw_fit.params['race_black'] * 100,
                         'max_weight': reg['ipw'].max()}])
ipw_row.to_csv(OUTPUT_DIR / 'ipw_missing_sensitivity.csv', index=False)
print(ipw_row.round(3).to_string(index=False))

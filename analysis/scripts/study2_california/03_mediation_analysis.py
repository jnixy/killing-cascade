"""
Study 2 - California: Mediation analysis.

Tests whether wound count mediates the relationship between officer count
and fatality (consistent with "contagious fire" hypothesis).

Causal pathway:
  More officers -> More wounds (hits) -> Higher fatality

Uses Baron & Kenny approach with OLS (linear probability model).
OLS mediation is cleaner than logistic: all paths on the same scale,
so Total - Direct = Indirect without the cross-model scaling issues
noted by Mood (2010).

Adapted from:
  - officer_count_paper/Study 2 - California/scripts/07_wound_count_mediation.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from lpm_utils import fit_lpm  # noqa: E402
MERGED_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = MERGED_ROOT / 'outputs' / 'study2'
TABLES_DIR = OUTPUT_DIR / 'tables'
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 70)
print('MERGED PAPER - STUDY 2: MEDIATION ANALYSIS')
print('Officers -> Wound Count -> Fatality')
print('Primary specification: OLS (linear probability model)')
print('=' * 70)

# =============================================================================
# LOAD DATA
# =============================================================================
# Same sample, covariates, and estimator as the primary CA models (02_run_analysis.py), so
# the decomposition's total effect equals Table 3 M4. Revised 2026-10-01 (PQ R&R): the
# Round 1 version used demographics only and a separate, larger sample.
df = pd.read_csv(OUTPUT_DIR / 'california_regression_data.csv')
print(f'Mediation analysis sample: {len(df):,} cases')

covariates = ['age_10yr', 'female', 'race_black', 'race_hispanic', 'armed_assault',
              'civ_fired_at_officer', 'civ_threat_firearm', 'officer_injured', 'civ_vehicle_against_officer',
              'weapon_firearm', 'weapon_replica', 'weapon_knife', 'weapon_other',
              'contact_stop', 'contact_planned', 'contact_other', 'multi_civilian']

# =============================================================================
# DESCRIPTIVE ANALYSIS
# =============================================================================
print('\n' + '=' * 70)
print('DESCRIPTIVE ANALYSIS')
print('=' * 70)

print('\n--- Wound Count Distribution ---')
print(df['wound_count'].describe())

print('\n--- Wound Count by Officer Count ---')
print(f'{"Officers":<12} {"Mean Wounds":<15} {"SD":<10} {"N":<8}')
print('-' * 45)
for n_off in [1, 2, 3, 4, 5, '6+']:
    if n_off == '6+':
        subset = df[df['num_involved_officers'] >= 6]
    else:
        subset = df[df['num_involved_officers'] == n_off]
    if len(subset) > 5:
        mean_w = subset['wound_count'].mean()
        sd_w = subset['wound_count'].std()
        print(f'{n_off:<12} {mean_w:<15.2f} {sd_w:<10.2f} {len(subset)}')

# Correlation
corr, p_corr = stats.pearsonr(df['num_involved_officers'], df['wound_count'])
print(f'\nCorrelation (officers vs wound count): r = {corr:.3f}, p = {p_corr:.6f}')

print('\n--- CFR by Wound Count ---')
print(f'{"Wounds":<12} {"CFR":<10} {"Fatal/N":<15}')
print('-' * 40)
for n_wounds in [1, 2, 3, 4, '5+']:
    if n_wounds == '5+':
        subset = df[df['wound_count'] >= 5]
    else:
        subset = df[df['wound_count'] == n_wounds]
    if len(subset) > 5:
        cfr = subset['fatal'].mean() * 100
        fatal_n = int(subset['fatal'].sum())
        print(f'{n_wounds:<12} {cfr:.1f}%     {fatal_n}/{len(subset)}')

# =============================================================================
# MEDIATION ANALYSIS (Baron & Kenny, OLS)
# =============================================================================
print('\n' + '=' * 70)
print('MEDIATION ANALYSIS (OLS)')
print('=' * 70)

y = df['fatal']

# Step 1: Total effect (X -> Y) — OLS
print('\n--- Step 1: Total Effect (Officers -> Fatality) ---')
X1 = sm.add_constant(df[['num_involved_officers'] + covariates])
# All paths include agency FE (revised 2026-10-02): the total is Table 4's M5, and the b path
# adds regions wounded to M5, matching M7's specification without wound location.
model_total = fit_lpm(df, 'fatal', ['num_involved_officers'] + covariates, agency_fe=True)
total_effect = model_total.params['num_involved_officers']
total_coef_pp = total_effect * 100
total_ci = (model_total.conf_int().loc['num_involved_officers', 0] * 100,
            model_total.conf_int().loc['num_involved_officers', 1] * 100)
total_p = model_total.pvalues['num_involved_officers']
print(f'Total effect: {total_coef_pp:.2f} pp ({total_ci[0]:.2f}-{total_ci[1]:.2f})')
print(f'p = {total_p:.6f}')

# Step 2: Effect of X on M (Officers -> Wound count) — OLS (unchanged)
print('\n--- Step 2: Officers -> Wound Count (a path) ---')
X2 = sm.add_constant(df[['num_involved_officers'] + covariates])
m = df['wound_count']
model_a = fit_lpm(df, 'wound_count', ['num_involved_officers'] + covariates, agency_fe=True)
a_path = model_a.params['num_involved_officers']
a_p = model_a.pvalues['num_involved_officers']
print(f'a path (officers -> wounds): {a_path:.4f}')
print(f'  Each additional officer associated with {a_path:.3f} more wound locations')
print(f'  p = {a_p:.6f}')

# Step 3: Effect of M on Y controlling for X (Wounds -> Fatality | Officers) — OLS
print('\n--- Step 3: Wound Count -> Fatality (b path, controlling for officers) ---')
X3 = sm.add_constant(df[['wound_count', 'num_involved_officers'] + covariates])
model_b = fit_lpm(df, 'fatal', ['wound_count', 'num_involved_officers'] + covariates, agency_fe=True)
b_path = model_b.params['wound_count']
b_coef_pp = b_path * 100
b_p = model_b.pvalues['wound_count']
print(f'b path: {b_coef_pp:.2f} pp per wound location')
print(f'  Each additional wound location associated with {b_coef_pp:.1f} pp higher fatality risk')
print(f'  p = {b_p:.6f}')

# Step 4: Direct effect (Officers -> Fatality, controlling for wounds) — OLS
print('\n--- Step 4: Direct Effect (Officers -> Fatality | Wound Count) ---')
direct_effect = model_b.params['num_involved_officers']
direct_coef_pp = direct_effect * 100
direct_ci = (model_b.conf_int().loc['num_involved_officers', 0] * 100,
             model_b.conf_int().loc['num_involved_officers', 1] * 100)
direct_p = model_b.pvalues['num_involved_officers']
print(f'Direct effect: {direct_coef_pp:.2f} pp ({direct_ci[0]:.2f}-{direct_ci[1]:.2f})')
print(f'p = {direct_p:.6f}')

# =============================================================================
# MEDIATION SUMMARY
# =============================================================================
print('\n' + '=' * 70)
print('MEDIATION SUMMARY')
print('=' * 70)

# In OLS, Total - Direct = Indirect (same scale, no cross-model scaling issue)
indirect_effect = total_effect - direct_effect
indirect_coef_pp = indirect_effect * 100
prop_mediated = indirect_effect / total_effect * 100 if total_effect != 0 else 0

print(f'\nTotal effect:    {total_coef_pp:.2f} pp')
print(f'Direct effect:   {direct_coef_pp:.2f} pp')
print(f'Indirect effect: {indirect_coef_pp:.2f} pp')
print(f'\nProportion mediated: {prop_mediated:.1f}%')

if prop_mediated > 0:
    print(f'\n{prop_mediated:.1f}% of the officer-count association is statistically accounted for by regions wounded.')
    print('Descriptive decomposition only: officer count and regions wounded may share')
    print('common causes (incident severity), so this is not a causal mediation estimate.')

if direct_coef_pp > 0 and direct_p < 0.05:
    print(f'\nThe direct effect remains significant ({direct_coef_pp:.2f} pp, p = {direct_p:.3f}).')
    print(f'Wound count partially mediates but does not fully explain the association.')
elif direct_coef_pp > 0:
    print(f'\nThe direct effect is attenuated and non-significant ({direct_coef_pp:.2f} pp, p = {direct_p:.3f}).')
    print(f'Wound count substantially mediates the officer count effect.')

# =============================================================================
# CATEGORICAL WOUND COUNT MODEL
# =============================================================================
print('\n' + '=' * 70)
print('MODEL WITH CATEGORICAL WOUND COUNT')
print('=' * 70)

df['wounds_2'] = (df['wound_count'] == 2).astype(int)
df['wounds_3'] = (df['wound_count'] == 3).astype(int)
df['wounds_4plus'] = (df['wound_count'] >= 4).astype(int)

X4 = sm.add_constant(df[['num_involved_officers', 'wounds_2', 'wounds_3', 'wounds_4plus'] + covariates])
model_cat = fit_lpm(df, 'fatal', ['num_involved_officers', 'wounds_2', 'wounds_3', 'wounds_4plus'] + covariates, agency_fe=True)

print('\nCoefficients (pp):')
print(f'  Officer count: {model_cat.params["num_involved_officers"]*100:.2f} pp')
print(f'  2 wounds vs 1: {model_cat.params["wounds_2"]*100:.2f} pp')
print(f'  3 wounds vs 1: {model_cat.params["wounds_3"]*100:.2f} pp')
print(f'  4+ wounds vs 1: {model_cat.params["wounds_4plus"]*100:.2f} pp')

# =============================================================================
# SAVE RESULTS
# =============================================================================
print('\n' + '=' * 70)
print('SAVING RESULTS')
print('=' * 70)

results = {
    'total_effect_Coef': total_coef_pp,
    'total_effect_CI_lower': total_ci[0],
    'total_effect_CI_upper': total_ci[1],
    'total_effect_p': total_p,
    'a_path': a_path,
    'a_path_p': a_p,
    'b_path_Coef': b_coef_pp,
    'b_path_p': b_p,
    'direct_effect_Coef': direct_coef_pp,
    'direct_effect_CI_lower': direct_ci[0],
    'direct_effect_CI_upper': direct_ci[1],
    'direct_effect_p': direct_p,
    'indirect_effect_Coef': indirect_coef_pp,
    'proportion_mediated': prop_mediated,
    'correlation_officers_wounds': corr,
    'correlation_p': p_corr,
    'N': len(df)
}

pd.DataFrame([results]).to_csv(TABLES_DIR / 'mediation_results.csv', index=False)
print('Saved: mediation_results.csv')

# Save wound count by officers
wound_by_officers = []
for n_off in range(1, 11):
    subset = df[df['num_involved_officers'] == n_off]
    if len(subset) >= 5:
        wound_by_officers.append({
            'officers': n_off,
            'mean_wounds': subset['wound_count'].mean(),
            'sd_wounds': subset['wound_count'].std(),
            'cfr': subset['fatal'].mean() * 100,
            'n': len(subset)
        })

pd.DataFrame(wound_by_officers).to_csv(TABLES_DIR / 'wound_count_by_officers.csv', index=False)
print('Saved: wound_count_by_officers.csv')

print('\n' + '=' * 70)
print('MEDIATION ANALYSIS COMPLETE')
print('=' * 70)

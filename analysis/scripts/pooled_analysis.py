"""
Pooled Analysis: Combined Texas + California OLS Regressions

Replaces the k=2 meta-analysis with actual pooled regressions on stacked
TX + CA data using common covariates available in both states.

Outputs:
  outputs/pooled/tables/descriptive_by_officer_count.csv
  outputs/pooled/tables/regression_models.csv
  outputs/pooled_regression_results.csv  (replaces meta_analysis_results.csv)

Primary specification (revised 2026-10-01, PQ R&R): OLS linear probability model with
state-by-year fixed effects and agency-clustered SEs, on the two state regression samples.

Author: Nix & Adams
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lpm_utils import fit_lpm, fe_dummies  # noqa: E402
MERGED_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = MERGED_ROOT / 'outputs' / 'pooled'
TABLES_DIR = OUTPUT_DIR / 'tables'
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 70)
print('POOLED ANALYSIS: COMBINED TEXAS + CALIFORNIA OLS')
print('Common covariates: officer count, age, gender, race, armed/resistance')
print('=' * 70)

# =============================================================================
# SECTION 1: LOAD AND HARMONIZE DATA
# =============================================================================
print('\n' + '=' * 70)
print('SECTION 1: LOADING AND HARMONIZING DATA')
print('=' * 70)

tx_raw = pd.read_csv(MERGED_ROOT / 'outputs' / 'study1' / 'texas_analysis_ready.csv')
ca_raw = pd.read_csv(MERGED_ROOT / 'outputs' / 'study2' / 'california_analysis_ready.csv')

# Pool the two state regression samples so the pooled N reconciles with Tables 2 and 3.
# Note the exposures differ: TX counts officers who discharged (one OAG report each); CA
# counts officers involved (URSUS NUM_INVOLVED_OFFICERS).
tx_pool = pd.read_csv(MERGED_ROOT / 'outputs' / 'study1' / 'texas_regression_data.csv')
ca_pool = pd.read_csv(MERGED_ROOT / 'outputs' / 'study2' / 'california_regression_data.csv')
tx_pool['state'] = 'Texas'
tx_pool['officer_count'] = tx_pool['Officer_Count']
ca_pool['state'] = 'California'
ca_pool['officer_count'] = ca_pool['num_involved_officers']
ca_pool['armed'] = ca_pool['armed_assault']

common_cols = ['state', 'fatal', 'officer_count', 'age_10yr', 'female',
               'race_black', 'race_hispanic', 'armed', 'race_std', 'age_numeric',
               'year', 'agency']

print(f'\nTexas pooled sample: {len(tx_pool):,} cases')
print(f'California pooled sample: {len(ca_pool):,} cases')

pooled = pd.concat([tx_pool[common_cols], ca_pool[common_cols]], ignore_index=True)
pooled['state_ca'] = (pooled['state'] == 'California').astype(int)
# Year effects are estimated separately by state (reporting regimes differ); agencies are
# prefixed by state so clusters never merge across states.
pooled['year'] = pooled['state'].str[:2] + '_' + pooled['year'].astype(int).astype(str)
pooled['agency'] = pooled['state'].str[:2] + '_' + pooled['agency'].astype(str)

print(f'Combined pooled sample: {len(pooled):,} cases')

# =============================================================================
# SECTION 2: DESCRIPTIVE STATISTICS BY OFFICER COUNT CATEGORY
# =============================================================================
print('\n' + '=' * 70)
print('SECTION 2: DESCRIPTIVE STATISTICS BY OFFICER COUNT')
print('=' * 70)

desc_rows = []
for state_name, state_df in [('Texas', tx_pool), ('California', ca_pool)]:
    # Use full-sample data (not regression-filtered) for descriptives
    state_full = tx_raw if state_name == 'Texas' else ca_raw
    oc_col = 'Officer_Count' if state_name == 'Texas' else 'num_involved_officers'
    state_full = state_full[state_full[oc_col].notna()]
    n_state = len(state_full)

    for cat_label, mask in [('1 officer', state_full[oc_col] == 1),
                            ('2 officers', state_full[oc_col] == 2),
                            ('3 officers', state_full[oc_col] == 3),
                            ('4+ officers', state_full[oc_col] >= 4)]:
        subset = state_full[mask]
        n = len(subset)
        fatal = int(subset['fatal'].sum())
        cfr = fatal / n * 100 if n > 0 else 0
        desc_rows.append({
            'state': state_name, 'category': cat_label,
            'n': n, 'pct_of_state': n / n_state * 100,
            'fatal': fatal, 'cfr': cfr
        })
        print(f'  {state_name} - {cat_label}: {n:,} ({n/n_state*100:.1f}%), '
              f'CFR = {cfr:.1f}%')

desc_df = pd.DataFrame(desc_rows)
desc_df.to_csv(TABLES_DIR / 'descriptive_by_officer_count.csv', index=False)
print(f'\nSaved: descriptive_by_officer_count.csv')

# =============================================================================
# SECTION 3: POOLED OLS REGRESSIONS
# =============================================================================
print('\n' + '=' * 70)
print('SECTION 3: POOLED OLS REGRESSIONS')
print('=' * 70)

y = pooled['fatal']
results_dict = {}

# --- PM1: Officer count + state FE ---
print('\n--- PM1: Officer Count + State FE ---')
pm1 = fit_lpm(pooled, 'fatal', ['officer_count'])

coef1 = pm1.params['officer_count'] * 100
ci1 = pm1.conf_int().loc['officer_count'] * 100
print(f'Officer count coef: {coef1:.2f} pp ({ci1[0]:.2f}-{ci1[1]:.2f}), '
      f'p = {pm1.pvalues["officer_count"]:.4f}')

results_dict['PM1: Officer Count + State'] = {
    'officer_count_Coef': coef1,
    'officer_count_CI_lower': ci1[0],
    'officer_count_CI_upper': ci1[1],
    'officer_count_p': pm1.pvalues['officer_count'],
    'R2': pm1.rsquared, 'N': len(pooled)
}

# --- PM2: + Demographics ---
print('\n--- PM2: + Demographics (Age, Gender, Race) ---')
pm2 = fit_lpm(pooled, 'fatal', ['officer_count', 'age_10yr', 'female', 'race_black', 'race_hispanic'])

coef2 = pm2.params['officer_count'] * 100
ci2 = pm2.conf_int().loc['officer_count'] * 100
print(f'Officer count coef: {coef2:.2f} pp ({ci2[0]:.2f}-{ci2[1]:.2f}), '
      f'p = {pm2.pvalues["officer_count"]:.4f}')

pm2_results = {
    'officer_count_Coef': coef2,
    'officer_count_CI_lower': ci2[0],
    'officer_count_CI_upper': ci2[1],
    'officer_count_p': pm2.pvalues['officer_count'],
}
for var in pm2.params.index:
    if var == 'const' or var.startswith('yr_'):
        continue
    pm2_results[f'{var}_Coef'] = pm2.params[var] * 100
    pm2_results[f'{var}_CI_lower'] = pm2.conf_int().loc[var, 0] * 100
    pm2_results[f'{var}_CI_upper'] = pm2.conf_int().loc[var, 1] * 100
    pm2_results[f'{var}_p'] = pm2.pvalues[var]
pm2_results['R2'] = pm2.rsquared
pm2_results['N'] = len(pooled)
results_dict['PM2: + Demographics'] = pm2_results

# --- PM3: Full common model (+ Armed) ---
print('\n--- PM3: Full Common Model (+ Armed/Resistance) ---')
pm3 = fit_lpm(pooled, 'fatal', ['officer_count', 'age_10yr', 'female', 'race_black', 'race_hispanic', 'armed'])

coef3 = pm3.params['officer_count'] * 100
ci3 = pm3.conf_int().loc['officer_count'] * 100
print(f'\nOfficer count coef: {coef3:.2f} pp ({ci3[0]:.2f}-{ci3[1]:.2f}), '
      f'p = {pm3.pvalues["officer_count"]:.4f}')
print(f'Black coef: {pm3.params["race_black"]*100:.2f} pp')
print(f'Hispanic coef: {pm3.params["race_hispanic"]*100:.2f} pp')

pm3_results = {}
for var in pm3.params.index:
    if var == 'const' or var.startswith('yr_'):
        continue
    pm3_results[f'{var}_Coef'] = pm3.params[var] * 100
    pm3_results[f'{var}_CI_lower'] = pm3.conf_int().loc[var, 0] * 100
    pm3_results[f'{var}_CI_upper'] = pm3.conf_int().loc[var, 1] * 100
    pm3_results[f'{var}_p'] = pm3.pvalues[var]
pm3_results['R2'] = pm3.rsquared
pm3_results['N'] = len(pooled)
results_dict['PM3: Full Common'] = pm3_results

# Save all pooled model results
all_models = []
for model_name, res in results_dict.items():
    row = {'Model': model_name}
    row.update(res)
    all_models.append(row)
pd.DataFrame(all_models).to_csv(TABLES_DIR / 'regression_models.csv', index=False)
print(f'\nSaved: pooled regression_models.csv')

# =============================================================================
# SECTION 4: POOLED LOGISTIC REGRESSIONS (for appendix)
# =============================================================================
print('\n' + '=' * 70)
print('SECTION 4: POOLED LOGISTIC REGRESSIONS (APPENDIX)')
print('=' * 70)

logit_results_dict = {}


def fit_pooled_logit(xvars):
    X = sm.add_constant(pd.concat([pooled[xvars].astype(float), fe_dummies(pooled, 'year', 'yr')], axis=1))
    groups = pd.factorize(pooled['agency'])[0]
    return sm.Logit(pooled['fatal'], X).fit(disp=0, cov_type='cluster', cov_kwds={'groups': groups})


# --- LM1: Officer count + state FE ---
print('\n--- LM1: Officer Count + State FE ---')
lm1 = fit_pooled_logit(['officer_count'])

or1 = np.exp(lm1.params['officer_count'])
ci1_logit = np.exp(lm1.conf_int().loc['officer_count'])
print(f'Officer count OR: {or1:.2f} ({ci1_logit[0]:.2f}-{ci1_logit[1]:.2f}), '
      f'p = {lm1.pvalues["officer_count"]:.4f}')

logit_results_dict['LM1: Officer Count + State'] = {
    'officer_count_OR': or1,
    'officer_count_CI_lower': ci1_logit[0],
    'officer_count_CI_upper': ci1_logit[1],
    'officer_count_p': lm1.pvalues['officer_count'],
    'Pseudo_R2': lm1.prsquared, 'N': len(pooled)
}

# --- LM2: + Demographics ---
print('\n--- LM2: + Demographics ---')
lm2 = fit_pooled_logit(['officer_count', 'age_10yr', 'female', 'race_black', 'race_hispanic'])

lm2_results = {}
for var in lm2.params.index:
    if var == 'const' or var.startswith('yr_'):
        continue
    lm2_results[f'{var}_OR'] = np.exp(lm2.params[var])
    lm2_results[f'{var}_CI_lower'] = np.exp(lm2.conf_int().loc[var, 0])
    lm2_results[f'{var}_CI_upper'] = np.exp(lm2.conf_int().loc[var, 1])
    lm2_results[f'{var}_p'] = lm2.pvalues[var]
lm2_results['Pseudo_R2'] = lm2.prsquared
lm2_results['N'] = len(pooled)
logit_results_dict['LM2: + Demographics'] = lm2_results

print(f'Officer count OR: {lm2_results["officer_count_OR"]:.2f} '
      f'({lm2_results["officer_count_CI_lower"]:.2f}-{lm2_results["officer_count_CI_upper"]:.2f})')

# --- LM3: Full common model (+ Armed) ---
print('\n--- LM3: Full Common Model ---')
lm3 = fit_pooled_logit(['officer_count', 'age_10yr', 'female', 'race_black', 'race_hispanic', 'armed'])

lm3_results = {}
for var in lm3.params.index:
    if var == 'const' or var.startswith('yr_'):
        continue
    lm3_results[f'{var}_OR'] = np.exp(lm3.params[var])
    lm3_results[f'{var}_CI_lower'] = np.exp(lm3.conf_int().loc[var, 0])
    lm3_results[f'{var}_CI_upper'] = np.exp(lm3.conf_int().loc[var, 1])
    lm3_results[f'{var}_p'] = lm3.pvalues[var]
lm3_results['Pseudo_R2'] = lm3.prsquared
lm3_results['N'] = len(pooled)
logit_results_dict['LM3: Full Common'] = lm3_results

print(f'Officer count OR: {lm3_results["officer_count_OR"]:.2f} '
      f'({lm3_results["officer_count_CI_lower"]:.2f}-{lm3_results["officer_count_CI_upper"]:.2f})')
print(f'Black OR: {lm3_results["race_black_OR"]:.2f}')

# Save logistic results
logit_rows = []
for model_name, res in logit_results_dict.items():
    row = {'Model': model_name}
    row.update(res)
    logit_rows.append(row)
pd.DataFrame(logit_rows).to_csv(TABLES_DIR / 'regression_models_logistic.csv', index=False)
print(f'\nSaved: pooled regression_models_logistic.csv')

# =============================================================================
# SECTION 5: SUMMARY CSV (replaces meta_analysis_results.csv)
# =============================================================================
print('\n' + '=' * 70)
print('SECTION 4: SUMMARY FOR MANUSCRIPT')
print('=' * 70)

# Read state-specific results for the summary table
tx_models = pd.read_csv(MERGED_ROOT / 'outputs' / 'study1' / 'tables' / 'regression_models.csv')
tx_full = tx_models[tx_models['Model'] == 'Model 3: + Weapon'].iloc[0]

ca_models = pd.read_csv(MERGED_ROOT / 'outputs' / 'study2' / 'tables' / 'regression_models.csv')
# CA M3 (demographics + armed/assault) is the specification comparable to TX M3: both are
# total associations, not adjusted for post-exposure wound variables.
ca_m5 = ca_models[ca_models['Model'] == 'M3: + Resistance'].iloc[0]

# Get CA officer count column name (may have spaces converted)
ca_off_coef_col = [c for c in ca_m5.index if 'Officers' in c and c.endswith('_Coef')][0]
ca_off_ci_lo_col = [c for c in ca_m5.index if 'Officers' in c and c.endswith('_CI_lower')][0]
ca_off_ci_hi_col = [c for c in ca_m5.index if 'Officers' in c and c.endswith('_CI_upper')][0]

summary = pd.DataFrame([
    {
        'study': 'Texas',
        'Coef': tx_full['Officer_Count_Coef'],
        'CI_lower': tx_full['Officer_Count_CI_lower'],
        'CI_upper': tx_full['Officer_Count_CI_upper'],
    },
    {
        'study': 'California',
        'Coef': ca_m5[ca_off_coef_col],
        'CI_lower': ca_m5[ca_off_ci_lo_col],
        'CI_upper': ca_m5[ca_off_ci_hi_col],
    },
    {
        'study': 'Pooled',
        'Coef': coef3,
        'CI_lower': ci3[0],
        'CI_upper': ci3[1],
    },
])
summary.to_csv(MERGED_ROOT / 'outputs' / 'pooled_regression_results.csv', index=False)

print(f'\nTexas (M3):      Coef = {tx_full["Officer_Count_Coef"]:.2f} pp '
      f'({tx_full["Officer_Count_CI_lower"]:.2f}-{tx_full["Officer_Count_CI_upper"]:.2f})')
print(f'California (M3): Coef = {ca_m5[ca_off_coef_col]:.2f} pp '
      f'({ca_m5[ca_off_ci_lo_col]:.2f}-{ca_m5[ca_off_ci_hi_col]:.2f})')
print(f'Pooled (PM3):    Coef = {coef3:.2f} pp ({ci3[0]:.2f}-{ci3[1]:.2f})')
print(f'\nPooled N = {len(pooled):,}')

print(f'\nSaved: pooled_regression_results.csv')

print('\n' + '=' * 70)
print('POOLED ANALYSIS COMPLETE')
print('=' * 70)

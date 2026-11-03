"""
Study 1 - Texas: Statistical models examining officer count and fatality.

Part A: Officer count and fatality (full sample)
Part B: Officer demographics and fatality (single-officer incidents only)
Part C: Race interaction tests

Primary specification: OLS (linear probability model) with HC1 robust SEs.
Logistic regression results saved separately for appendix tables.

Adapted from officer_count_paper/Study 1 - Texas/scripts/02_run_analysis.py
and 03_interaction_effects.py for the merged manuscript (Nix & Adams).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from pathlib import Path
import os
import sys
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from lpm_utils import fit_lpm, fe_dummies, cluster_summary  # noqa: E402
MERGED_ROOT = SCRIPT_DIR.parent.parent
# TX_OUTPUT_DIR: see 01_prepare_data.py (used by revision/18_tx_linkage_sensitivity.py)
OUTPUT_DIR = Path(os.environ.get('TX_OUTPUT_DIR', MERGED_ROOT / 'outputs' / 'study1'))
TABLES_DIR = OUTPUT_DIR / 'tables'
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 70)
print('MERGED PAPER - STUDY 1: STATISTICAL MODELS')
print('Texas Justice Initiative OIS Data')
print('Primary specification: OLS (linear probability model)')
print('=' * 70)

# =============================================================================
# SECTION 1: LOAD DATA AND PREPARE REGRESSION SAMPLE
# =============================================================================
print('\n' + '=' * 70)
print('SECTION 1: LOADING DATA')
print('=' * 70)

df = pd.read_csv(OUTPUT_DIR / 'texas_analysis_ready.csv')
print(f'Loaded {len(df):,} OIS cases')

df_with_count = df[df['Officer_Count'].notna()].copy()
print(f'Cases with officer count data: {len(df_with_count):,}')

# Prepare regression sample
reg_df = df_with_count[df_with_count['race_std'].isin(['White', 'Black', 'Hispanic'])].copy()
reg_df = reg_df.dropna(subset=['age_numeric', 'Victim_Gender', 'Officer_Count'])
reg_df = reg_df[reg_df['Victim_Gender'].isin(['MALE', 'FEMALE', 'Male', 'Female'])]

print(f'\nRegression sample: {len(reg_df):,} cases')
print(f'  (White/Black/Hispanic with complete data on age, gender, officer count)')

# Create dummy variables
reg_df['race_black'] = (reg_df['race_std'] == 'Black').astype(int)
reg_df['race_hispanic'] = (reg_df['race_std'] == 'Hispanic').astype(int)
reg_df['female'] = reg_df['Victim_Gender'].str.upper().eq('FEMALE').astype(int)
reg_df['age_10yr'] = reg_df['age_numeric'] / 10
reg_df['armed'] = reg_df['had_weapon'].astype(int)
reg_df['log_officers'] = np.log(reg_df['Officer_Count'])
reg_df['off_duty'] = reg_df['officer_off_duty'].astype(int)
reg_df['officer_female_reg'] = reg_df['officer_female'].astype(int)
reg_df['officer_black_reg'] = reg_df['officer_black'].astype(int)
reg_df['officer_hispanic_reg'] = reg_df['officer_hispanic'].astype(int)

# =============================================================================
# SECTION 2: DESCRIPTIVE STATISTICS
# =============================================================================
print('\n' + '=' * 70)
print('SECTION 2: DESCRIPTIVE STATISTICS')
print('=' * 70)

print(f'\nOfficer count distribution:')
print(f'  Mean: {df_with_count["Officer_Count"].mean():.2f}')
print(f'  Median: {df_with_count["Officer_Count"].median():.1f}')
print(f'  SD: {df_with_count["Officer_Count"].std():.2f}')
print(f'  Range: {df_with_count["Officer_Count"].min():.0f} - {df_with_count["Officer_Count"].max():.0f}')

# CFR by officer count category
print('\nCase Fatality Rate by Officer Count Category:')
cfr_by_cat = []
for cat in ['1 officer', '2 officers', '3 officers', '4-5 officers', '6+ officers']:
    subset = df_with_count[df_with_count['officer_count_cat'] == cat]
    if len(subset) > 0:
        fatal = subset['fatal'].sum()
        total = len(subset)
        cfr = fatal / total * 100
        print(f'  {cat}: {fatal}/{total} = {cfr:.1f}%')
        cfr_by_cat.append({'category': cat, 'fatal': fatal, 'total': total, 'cfr': cfr})

# Save descriptive stats for manuscript inline code
desc_stats = {
    'n_total': len(df),
    'n_with_count': len(df_with_count),
    'n_regression': len(reg_df),
    'pct_fatal': df_with_count['fatal'].mean() * 100,
    'pct_male': (reg_df['female'] == 0).mean() * 100,
    'mean_age': reg_df['age_numeric'].mean(),
    'sd_age': reg_df['age_numeric'].std(),
    'pct_white': (reg_df['race_std'] == 'White').mean() * 100,
    'pct_black': (reg_df['race_std'] == 'Black').mean() * 100,
    'pct_hispanic': (reg_df['race_std'] == 'Hispanic').mean() * 100,
    'pct_armed': reg_df['armed'].mean() * 100,
    'mean_officers': reg_df['Officer_Count'].mean(),
    'sd_officers': reg_df['Officer_Count'].std(),
    'cfr_single': df_with_count[df_with_count['Officer_Count'] == 1]['fatal'].mean() * 100,
    'cfr_multiple': df_with_count[df_with_count['Officer_Count'] >= 2]['fatal'].mean() * 100,
}
pd.DataFrame([desc_stats]).to_csv(TABLES_DIR / 'descriptive_stats.csv', index=False)
print(f'\nSaved: descriptive_stats.csv')

# =============================================================================
# PART A: OFFICER COUNT AND FATALITY (OLS — PRIMARY SPECIFICATION)
# =============================================================================
# Revised 2026-10-01 (PQ R&R): incident-year fixed effects and agency-clustered SEs in every
# model (see lpm_utils.py), and a new M4 adding incident circumstances.
print('\n' + '=' * 70)
print('PART A: OFFICER COUNT AND FATALITY (OLS)')
print('=' * 70)

# One 2014 incident (reported under the 2015 statute) shares the 2015 year effect.
reg_df['year'] = reg_df['incident_year'].clip(lower=2015)
reg_df['agency'] = reg_df['agency_name'].astype(str).str.upper().str.strip()

# Incident circumstances. Reference categories: emergency call; weapon missing in media.
# (OAG incident_result_of yields only four populated call categories in the sample.)
reg_df['call_traffic'] = (reg_df['incident_type_cat'] == 'Traffic Stop').astype(int)
reg_df['call_warrant'] = (reg_df['incident_type_cat'] == 'Warrant Service').astype(int)
reg_df['call_other'] = (reg_df['incident_type_cat'] == 'Other').astype(int)
for cat, col in [('Firearm', 'weapon_firearm'), ('Knife/cutting', 'weapon_knife'),
                 ('Vehicle', 'weapon_vehicle'), ('Other', 'weapon_other')]:
    reg_df[col] = (reg_df['weapon_media_cat'] == cat).astype(int)

circumstance_vars = ['officer_shot_by_civ', 'weapon_firearm', 'weapon_knife', 'weapon_vehicle',
                     'weapon_other', 'call_traffic', 'call_warrant', 'call_other',
                     'multi_civilian']

results_dict = {}
y = reg_df['fatal']


def collect(model, label):
    out = {}
    for var in model.params.index:
        if var == 'const' or var.startswith(('yr_', 'ag_')):
            continue
        out[f'{var}_Coef'] = model.params[var] * 100
        out[f'{var}_CI_lower'] = model.conf_int().loc[var, 0] * 100
        out[f'{var}_CI_upper'] = model.conf_int().loc[var, 1] * 100
        out[f'{var}_p'] = model.pvalues[var]
    out['R2'] = model.rsquared
    out['N'] = int(model.nobs)
    out['n_clusters'] = model.n_clusters
    coef = model.params['Officer_Count'] * 100
    ci = model.conf_int().loc['Officer_Count'] * 100
    print(f'{label}: officer = {coef:.2f} pp ({ci[0]:.2f}-{ci[1]:.2f}), '
          f'p = {model.pvalues["Officer_Count"]:.4f}, R2 = {model.rsquared:.3f}')
    return out


m1_vars = ['Officer_Count']
m2_vars = m1_vars + ['age_10yr', 'female', 'race_black', 'race_hispanic']
m3_vars = m2_vars + ['armed']
m4_vars = m3_vars + circumstance_vars
model_specs = [('Model 1: Unadjusted', m1_vars), ('Model 2: + Demographics', m2_vars),
               ('Model 3: + Weapon', m3_vars), ('Model 4: + Incident Circumstances', m4_vars)]
fitted = {}
for label, xvars in model_specs:
    fitted[label] = fit_lpm(reg_df, 'fatal', xvars)
    results_dict[label] = collect(fitted[label], label)

# M5 adds agency fixed effects to M4 (revised 2026-10-02; was labeled "Model 4 + Agency FE")
fitted['Model 5: + Agency FE'] = fit_lpm(reg_df, 'fatal', m4_vars, agency_fe=True)
results_dict['Model 5: + Agency FE'] = collect(fitted['Model 5: + Agency FE'], 'Model 5: + Agency FE')
# Round 1 specification retained for the before/after comparison
fitted['Model 3 (HC1, no FE)'] = fit_lpm(reg_df, 'fatal', m3_vars, year_fe=False, cov='HC1')
results_dict['Model 3 (HC1, no FE)'] = collect(fitted['Model 3 (HC1, no FE)'], 'Model 3 (HC1, no FE)')

model1, model2, model3, model4 = (fitted[label] for label, _ in model_specs)

clusters = cluster_summary(reg_df)
pd.DataFrame([clusters]).to_csv(TABLES_DIR / 'cluster_summary.csv', index=False)
print(f'Agencies: {clusters}')

all_models = []
for model_name, res in results_dict.items():
    row = {'Model': model_name}
    row.update(res)
    all_models.append(row)
pd.DataFrame(all_models).to_csv(TABLES_DIR / 'regression_models.csv', index=False)
print(f'\nSaved: regression_models.csv')

# =============================================================================
# PART A-LOGISTIC: LOGISTIC REGRESSION (FOR APPENDIX)
# =============================================================================
print('\n' + '=' * 70)
print('PART A-LOGISTIC: LOGISTIC REGRESSION (FOR APPENDIX)')
print('=' * 70)

logistic_dict = {}
for label, xvars in model_specs:
    d = reg_df.dropna(subset=['fatal'] + xvars)
    X = sm.add_constant(pd.concat([d[xvars].astype(float), fe_dummies(d, 'year', 'yr')], axis=1))
    groups = pd.factorize(d['agency'])[0]
    lm = sm.Logit(d['fatal'], X).fit(disp=0, cov_type='cluster', cov_kwds={'groups': groups})
    ame = lm.get_margeff(at='overall').summary_frame().loc['Officer_Count']
    res = {'Officer_AME': ame.iloc[0] * 100, 'Officer_AME_CI_lower': ame.iloc[4] * 100,
           'Officer_AME_CI_upper': ame.iloc[5] * 100}
    for var in xvars:
        res[f'{var}_OR'] = np.exp(lm.params[var])
        res[f'{var}_CI_lower'] = np.exp(lm.conf_int().loc[var, 0])
        res[f'{var}_CI_upper'] = np.exp(lm.conf_int().loc[var, 1])
        res[f'{var}_p'] = lm.pvalues[var]
    res['Pseudo_R2'] = lm.prsquared
    res['N'] = int(lm.nobs)
    logistic_dict[label] = res
    print(f'Logit {label}: officer AME = {res["Officer_AME"]:.2f} pp')

logistic_rows = []
for model_name, res in logistic_dict.items():
    row = {'Model': model_name}
    row.update(res)
    logistic_rows.append(row)
pd.DataFrame(logistic_rows).to_csv(TABLES_DIR / 'regression_models_logistic.csv', index=False)
print('Saved: regression_models_logistic.csv')

# =============================================================================
# PART B: OFFICER DEMOGRAPHICS (SINGLE-OFFICER INCIDENTS)
# =============================================================================
print('\n' + '=' * 70)
print('PART B: OFFICER DEMOGRAPHICS AND FATALITY')
print('(Single-Officer Incidents Only)')
print('=' * 70)

single_officer_df = reg_df[reg_df['Officer_Count'] == 1].copy()
print(f'\nSingle-officer incidents: {len(single_officer_df):,} cases')
print(f'  Fatal: {single_officer_df["fatal"].sum():,} ({single_officer_df["fatal"].mean()*100:.1f}%)')

y_single = single_officer_df['fatal']

# Full model with officer demographics (OLS)
print('\n--- Officer Demographics Model (OLS, Fully Adjusted) ---')
model_b = fit_lpm(single_officer_df, 'fatal', ['off_duty', 'officer_female_reg',
                                                'officer_black_reg', 'officer_hispanic_reg',
                                                'age_10yr', 'female', 'race_black',
                                                'race_hispanic', 'armed'])

officer_chars_data = []
for var, label in [('off_duty', 'Off-duty (vs On-duty)'),
                   ('officer_female_reg', 'Female officer (vs Male)'),
                   ('officer_black_reg', 'Black officer (vs White)'),
                   ('officer_hispanic_reg', 'Hispanic officer (vs White)')]:
    coef_val = model_b.params[var] * 100
    ci_l = model_b.conf_int().loc[var, 0] * 100
    ci_h = model_b.conf_int().loc[var, 1] * 100
    p_val = model_b.pvalues[var]
    sig = '*' if p_val < 0.05 else ''
    print(f'  {label}: Coef = {coef_val:.2f} pp ({ci_l:.2f}-{ci_h:.2f}), p = {p_val:.4f}{sig}')
    officer_chars_data.append({
        'Variable': label, 'Coef': coef_val, 'CI_lower': ci_l, 'CI_upper': ci_h, 'p_value': p_val
    })

pd.DataFrame(officer_chars_data).to_csv(TABLES_DIR / 'officer_characteristics.csv', index=False)
print(f'\nSaved: officer_characteristics.csv')

# =============================================================================
# PART C: RACE INTERACTION TESTS
# =============================================================================
print('\n' + '=' * 70)
print('PART C: RACE INTERACTION TESTS')
print('=' * 70)

# Race-stratified analysis (OLS)
main_races = ['White', 'Black', 'Hispanic']
stratified_results = []

for race in main_races:
    subset = reg_df[reg_df['race_std'] == race].copy()
    n = len(subset)
    fatal = subset['fatal'].sum()
    print(f'\n--- {race} (n={n:,}, {fatal} fatal) ---')

    if n < 50:
        print(f'  Insufficient sample size for regression')
        continue

    try:
        model = fit_lpm(subset, 'fatal', ['Officer_Count', 'age_10yr', 'female', 'armed'])
        coef_officers = model.params['Officer_Count'] * 100
        ci_low = model.conf_int().loc['Officer_Count', 0] * 100
        ci_high = model.conf_int().loc['Officer_Count', 1] * 100
        p_val = model.pvalues['Officer_Count']
        print(f'  Officer count coef: {coef_officers:.2f} pp ({ci_low:.2f}-{ci_high:.2f}), p = {p_val:.4f}')
        stratified_results.append({
            'race': race, 'n': n, 'fatal': fatal, 'Coef': coef_officers,
            'CI_lower': ci_low, 'CI_upper': ci_high, 'p_value': p_val
        })
    except Exception as e:
        print(f'  Model failed: {e}')

pd.DataFrame(stratified_results).to_csv(TABLES_DIR / 'officer_count_by_race_stratified.csv', index=False)

# Formal interaction test (F-test for OLS)
print('\n--- F-test for Race x Officer Count Interaction ---')

reg_df['officer_x_black'] = reg_df['Officer_Count'] * reg_df['race_black']
reg_df['officer_x_hispanic'] = reg_df['Officer_Count'] * reg_df['race_hispanic']

model_int = fit_lpm(reg_df, 'fatal', m4_vars + ['officer_x_black', 'officer_x_hispanic'])

# F-test on joint significance of interaction terms
f_test = model_int.f_test('officer_x_black = 0, officer_x_hispanic = 0')
f_stat = float(f_test.fvalue)
p_f = float(f_test.pvalue)
print(f'  F statistic: {f_stat:.3f}, df = 2, p = {p_f:.4f}')

if p_f < 0.05:
    print(f'  Result: Significant interaction - effect differs by race')
else:
    print(f'  Result: No significant interaction - effect is similar across races')

int_results = pd.DataFrame([{
    'test': 'F-test', 'statistic': f_stat, 'df': 2, 'p_value': p_f
}])
int_results.to_csv(TABLES_DIR / 'interaction_test_results.csv', index=False)

# =============================================================================
# SAVE REGRESSION DATA AND CFR TABLES
# =============================================================================
print('\n' + '=' * 70)
print('SAVING ALL RESULTS')
print('=' * 70)

reg_df.to_csv(OUTPUT_DIR / 'texas_regression_data.csv', index=False)
print(f'Saved: texas_regression_data.csv ({len(reg_df):,} cases)')

single_officer_df.to_csv(OUTPUT_DIR / 'texas_single_officer_data.csv', index=False)
print(f'Saved: texas_single_officer_data.csv ({len(single_officer_df):,} cases)')

# CFR by officer count for plotting
cfr_data = []
for n_officers in range(1, 11):
    subset = df_with_count[df_with_count['Officer_Count'] == n_officers]
    if len(subset) >= 5:
        cfr_data.append({
            'officers': n_officers, 'fatal': subset['fatal'].sum(),
            'total': len(subset), 'cfr': subset['fatal'].mean() * 100
        })
pd.DataFrame(cfr_data).to_csv(TABLES_DIR / 'cfr_by_officer_count.csv', index=False)

print('\n' + '=' * 70)
print('STUDY 1 ANALYSIS COMPLETE')
print('=' * 70)

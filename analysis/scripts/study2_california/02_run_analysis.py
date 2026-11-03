"""
Study 2 - California: Main statistical analysis.

Merged analysis combining wound location models (from wound_race_paper)
with officer count models (from officer_count_paper).

Sequential OLS (linear probability model) with HC1 robust SEs:
  M1: Officer count only (unadjusted)
  M2: + Demographics (age, gender, race)
  M3: + Resistance/weapon
  M4: + Wound location (CA-unique data)
  M5: + Wound count    (CA-unique data)

Plus: Descriptive statistics, differential aiming test, race interaction
tests (F-test), subgroup analyses.

Logistic regression results saved separately for appendix tables.

Adapted from:
  - wound_race_paper/scripts/run_full_analysis.py
  - officer_count_paper/Study 2 - California/scripts/02_run_analysis.py
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
from lpm_utils import fit_lpm, fe_dummies, cluster_summary  # noqa: E402
MERGED_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = MERGED_ROOT / 'outputs' / 'study2'
TABLES_DIR = OUTPUT_DIR / 'tables'
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 70)
print('MERGED PAPER - STUDY 2: MAIN ANALYSIS')
print('California DOJ URSUS Data')
print('Primary specification: OLS (linear probability model)')
print('=' * 70)

# =============================================================================
# LOAD DATA
# =============================================================================
df = pd.read_csv(OUTPUT_DIR / 'california_analysis_ready.csv')
print(f'\nLoaded {len(df):,} gunshot cases')

# =============================================================================
# PART A: DESCRIPTIVE STATISTICS
# =============================================================================
print('\n' + '=' * 70)
print('PART A: DESCRIPTIVE STATISTICS')
print('=' * 70)

n_total = len(df)
n_fatal = int(df['fatal'].sum())
cfr = df['fatal'].mean() * 100
n_male = (df['gender_std'] == 'Male').sum()
n_female = (df['gender_std'] == 'Female').sum()
mean_age = df['age_numeric'].mean()
std_age = df['age_numeric'].std()

print(f'\nTotal gunshot cases: {n_total:,}')
print(f'Fatal: {n_fatal:,} ({cfr:.1f}%)')
print(f'Male: {n_male:,} ({n_male/n_total*100:.1f}%)')
print(f'Female: {n_female:,} ({n_female/n_total*100:.1f}%)')
print(f'Age: mean = {mean_age:.1f}, SD = {std_age:.1f}')

# Race distribution
print('\nRace distribution:')
race_counts = df['race_std'].value_counts()
for race, count in race_counts.items():
    pct = count / n_total * 100
    print(f'  {race}: {count:,} ({pct:.1f}%)')

# Wound location distribution
print('\nWound location distribution:')
wound_counts = df['wound_location_std'].value_counts()
for loc, count in wound_counts.items():
    pct = count / n_total * 100
    print(f'  {loc}: {count:,} ({pct:.1f}%)')

# Officer count distribution
print('\nOfficer count distribution:')
df_with_count = df[df['num_involved_officers'].notna()]
print(f'  Cases with data: {len(df_with_count):,}')
print(f'  Mean: {df_with_count["num_involved_officers"].mean():.2f}')
print(f'  Median: {df_with_count["num_involved_officers"].median():.1f}')
print(f'  SD: {df_with_count["num_involved_officers"].std():.2f}')
print(f'  Range: {df_with_count["num_involved_officers"].min():.0f} - '
      f'{df_with_count["num_involved_officers"].max():.0f}')

# CFR by race
print('\nCase Fatality Rate by Race:')
main_races = ['White', 'Black', 'Hispanic']
df_main = df[df['race_std'].isin(main_races)]
for race in main_races:
    subset = df_main[df_main['race_std'] == race]
    fatal = subset['fatal'].sum()
    n = len(subset)
    rate = subset['fatal'].mean() * 100
    print(f'  {race}: {fatal}/{n} = {rate:.1f}%')

# CFR by wound location
wound_locs = ['Head/Neck', 'Chest', 'Abdomen', 'Back', 'Extremities']
print('\nCase Fatality Rate by Wound Location:')
for loc in wound_locs:
    subset = df[df['wound_location_std'] == loc]
    n = len(subset)
    fatal = subset['fatal'].sum()
    rate = subset['fatal'].mean() * 100 if n > 0 else 0
    print(f'  {loc}: {fatal}/{n} = {rate:.1f}%')

# CFR by officer count
print('\nCase Fatality Rate by Officer Count:')
for n_off in range(1, 8):
    subset = df_with_count[df_with_count['num_involved_officers'] == n_off]
    if len(subset) >= 10:
        fatal = subset['fatal'].sum()
        total = len(subset)
        cfr_oc = fatal / total * 100
        print(f'  {n_off} officer(s): {fatal}/{total} = {cfr_oc:.1f}%')

# =============================================================================
# PART B: DIFFERENTIAL AIMING TEST
# =============================================================================
print('\n' + '=' * 70)
print('PART B: RACE x WOUND LOCATION INDEPENDENCE TEST')
print('=' * 70)

# Same base as the regression sample and Table A5 (revised 2026-10-01)
df_chi = df_main[df_main['wound_location_std'].isin(wound_locs)
                 & df_main['age_numeric'].notna()
                 & df_main['gender_std'].isin(['Male', 'Female'])
                 & df_main['num_involved_officers'].notna()]
contingency_wounds = pd.crosstab(df_chi['race_std'], df_chi['wound_location_std'])
print('\nContingency table (Race x Wound Location):')
print(contingency_wounds)

chi2_wounds, p_wounds, dof_wounds, _ = stats.chi2_contingency(contingency_wounds)
print(f'\nChi-square: {chi2_wounds:.2f}, df = {dof_wounds}, p = {p_wounds:.3f}')
if p_wounds > 0.05:
    print('Result: Wound location does NOT differ significantly by race')
else:
    print('Result: Wound location differs significantly by race')

# =============================================================================
# PART C: SEQUENTIAL OLS MODELS (PRIMARY SPECIFICATION)
# =============================================================================
print('\n' + '=' * 70)
print('PART C: SEQUENTIAL OLS MODELS (LINEAR PROBABILITY)')
print('=' * 70)

# Prepare regression data
reg_df = df_main.copy()
reg_df = reg_df[reg_df['wound_location_std'].isin(wound_locs)]
reg_df = reg_df.dropna(subset=['age_numeric', 'gender_std', 'num_involved_officers'])
reg_df = reg_df[reg_df['gender_std'] != 'Unknown']

print(f'\nRegression sample: {len(reg_df):,} cases')
print(f'  (White/Black/Hispanic with complete data on age, gender, '
      f'wound location, officer count)')

# Create dummy variables
reg_df['race_black'] = (reg_df['race_std'] == 'Black').astype(int)
reg_df['race_hispanic'] = (reg_df['race_std'] == 'Hispanic').astype(int)
reg_df['female'] = (reg_df['gender_std'] == 'Female').astype(int)
reg_df['age_10yr'] = reg_df['age_numeric'] / 10
reg_df['armed_assault'] = reg_df['resistance_std'].isin(
    ['Armed/Deadly Force', 'Physical Assault']).astype(int)
reg_df['wound_head_neck'] = (reg_df['wound_location_std'] == 'Head/Neck').astype(int)
reg_df['wound_chest'] = (reg_df['wound_location_std'] == 'Chest').astype(int)
reg_df['wound_abdomen'] = (reg_df['wound_location_std'] == 'Abdomen').astype(int)
reg_df['wound_back'] = (reg_df['wound_location_std'] == 'Back').astype(int)

# Incident circumstances. Reference categories: weapon none/unknown, call/crime-in-progress
# contact, single civilian.
reg_df['weapon_firearm'] = (reg_df['weapon_type'] == 'Firearm').astype(int)
reg_df['weapon_replica'] = (reg_df['weapon_type'] == 'Replica firearm').astype(int)
reg_df['weapon_knife'] = (reg_df['weapon_type'] == 'Knife/blade').astype(int)
reg_df['weapon_other'] = (reg_df['weapon_type'] == 'Other weapon').astype(int)
reg_df['contact_stop'] = (reg_df['contact_reason_cat'] == 'Traffic/pedestrian stop').astype(int)
reg_df['contact_planned'] = (reg_df['contact_reason_cat'] == 'Planned/warrant').astype(int)
reg_df['contact_other'] = (reg_df['contact_reason_cat'] == 'Other/Unknown').astype(int)
reg_df['multi_civilian'] = (reg_df['num_involved_civilians'].fillna(1) > 1).astype(int)
for col in ['civ_fired_at_officer', 'civ_threat_firearm', 'officer_injured', 'civ_vehicle_against_officer']:
    reg_df[col] = reg_df[col].fillna(0).astype(int)

reg_df['year'] = reg_df['data_year']
reg_df['agency'] = reg_df['ori']

y = reg_df['fatal']
results_dict = {}

circumstance_vars = ['civ_fired_at_officer', 'civ_threat_firearm', 'officer_injured', 'civ_vehicle_against_officer',
                     'weapon_firearm', 'weapon_replica', 'weapon_knife', 'weapon_other',
                     'contact_stop', 'contact_planned', 'contact_other', 'multi_civilian']
wound_loc_vars = ['wound_head_neck', 'wound_chest', 'wound_abdomen', 'wound_back']


def extract_results(model, var_names, model_name):
    """Extract coefficients (in pp) and CIs for all variables in a model."""
    res = {'Model': model_name, 'N': int(model.nobs), 'R2': model.rsquared,
           'n_clusters': model.n_clusters}
    for var in var_names:
        coef_val = model.params[var] * 100
        ci_low = model.conf_int().loc[var, 0] * 100
        ci_high = model.conf_int().loc[var, 1] * 100
        p_val = model.pvalues[var]
        clean = var.replace('race_', '').replace('wound_', '').replace('_', ' ').title()
        res[f'{clean}_Coef'] = coef_val
        res[f'{clean}_CI_lower'] = ci_low
        res[f'{clean}_CI_upper'] = ci_high
        res[f'{clean}_p'] = p_val
    return res


# Model sequence (revised 2026-10-02): M1-M4 use incident-year FE + agency-clustered SEs; M5
# adds agency fixed effects, and the wound models (M6, M7) keep them, so the wound results
# compare civilians shot by the same agency. See lpm_utils.py.
m1_vars = ['num_involved_officers']
m2_vars = m1_vars + ['age_10yr', 'female', 'race_black', 'race_hispanic']
m3_vars = m2_vars + ['armed_assault']
m4_vars = m3_vars + circumstance_vars
wound_vars = m4_vars + wound_loc_vars          # M6
regions_vars = wound_vars + ['wound_count']    # M7
model_specs = [
    ('Model 1', m1_vars, 'M1: Officer Count', False),
    ('Model 2', m2_vars, 'M2: + Demographics', False),
    ('Model 3', m3_vars, 'M3: + Resistance', False),
    ('Model 4', m4_vars, 'M4: + Incident Circumstances', False),
    ('Model 5', m4_vars, 'M5: + Agency FE', True),
    ('Model 6', wound_vars, 'M6: + Wound Location', True),
    ('Model 7', regions_vars, 'M7: + Regions Wounded', True),
]
models_fitted = {}
for key, xvars, label, agency_fe in model_specs:
    fit = fit_lpm(reg_df, 'fatal', xvars, agency_fe=agency_fe)
    models_fitted[key] = (fit, xvars)
    results_dict[key] = extract_results(fit, xvars, label)
    print(f'{label}: officer = {fit.params["num_involved_officers"]*100:.2f} pp '
          f'(p = {fit.pvalues["num_involved_officers"]:.4f}), R2 = {fit.rsquared:.3f}')

# Wound models without agency FE (year FE only), for comparison with the earlier Round 2 draft
for key, xvars, label in [('Model 6 (year FE only)', wound_vars, 'M6 year FE only'),
                          ('Model 7 (year FE only)', regions_vars, 'M7 year FE only')]:
    fit = fit_lpm(reg_df, 'fatal', xvars)
    models_fitted[key] = (fit, xvars)
    results_dict[key] = extract_results(fit, xvars, label)
    print(f'{label}: officer = {fit.params["num_involved_officers"]*100:.2f} pp, R2 = {fit.rsquared:.3f}')

# Round 1 specification (HC1, no FE) retained for the before/after comparison
for key, xvars, label in [('Model 3 (HC1, no FE)', m3_vars, 'M3 HC1 no FE'),
                          ('Model 6 (HC1, no FE)', wound_vars, 'M6 HC1 no FE')]:
    fit = fit_lpm(reg_df, 'fatal', xvars, year_fe=False, cov='HC1')
    models_fitted[key] = (fit, xvars)
    results_dict[key] = extract_results(fit, xvars, label)

clusters = cluster_summary(reg_df)
pd.DataFrame([clusters]).to_csv(TABLES_DIR / 'cluster_summary.csv', index=False)
print(f'Agencies: {clusters}')

model1, model2, model3, model4, model5, model6, model7 = (models_fitted[f'Model {i}'][0] for i in range(1, 8))

# =============================================================================
# PART C-LOGISTIC: LOGISTIC REGRESSION (FOR APPENDIX)
# =============================================================================
print('\n' + '=' * 70)
print('PART C-LOGISTIC: LOGISTIC REGRESSION (FOR APPENDIX)')
print('=' * 70)


def fit_logit(data, xvars):
    """Logit with year FE and agency-clustered SEs; returns (result, officer AME row)."""
    d = data.dropna(subset=['fatal'] + xvars)
    X = sm.add_constant(pd.concat([d[xvars].astype(float), fe_dummies(d, 'year', 'yr')], axis=1))
    groups = pd.factorize(d['agency'])[0]
    res = sm.Logit(d['fatal'], X).fit(disp=0, cov_type='cluster', cov_kwds={'groups': groups})
    me = res.get_margeff(at='overall').summary_frame()
    return res, me.loc['num_involved_officers']


def extract_logistic_results(model, ame, var_names, model_name):
    """Extract ORs and CIs for all variables, plus the officer-count AME (pp)."""
    res = {'Model': model_name, 'N': int(model.nobs), 'Pseudo_R2': model.prsquared,
           'Officer_AME': ame.iloc[0] * 100, 'Officer_AME_CI_lower': ame.iloc[4] * 100,
           'Officer_AME_CI_upper': ame.iloc[5] * 100}
    for var in var_names:
        or_val = np.exp(model.params[var])
        ci_low = np.exp(model.conf_int().loc[var, 0])
        ci_high = np.exp(model.conf_int().loc[var, 1])
        p_val = model.pvalues[var]
        clean = var.replace('race_', '').replace('wound_', '').replace('_', ' ').title()
        res[f'{clean}_OR'] = or_val
        res[f'{clean}_CI_lower'] = ci_low
        res[f'{clean}_CI_upper'] = ci_high
        res[f'{clean}_p'] = p_val
    return res


# Logits keep incident-year FE only. A fixed-effects (conditional) logit discards every agency
# whose cases all died or all survived (125 agencies, 210 cases here) and does not support
# agency-clustered SEs, so the agency-FE step (M5) has no logit counterpart and the wound
# logits (M6, M7) are estimated without agency FE.
logistic_dict = {}
for key, xvars, label, agency_fe in model_specs:
    if key == 'Model 5':
        continue
    lres, ame = fit_logit(reg_df, xvars)
    logistic_dict[key] = extract_logistic_results(lres, ame, xvars, label)
    print(f'Logit {label}: officer AME = {ame.iloc[0]*100:.2f} pp')

logistic_rows = [{'Model': k, **v} for k, v in logistic_dict.items()]
pd.DataFrame(logistic_rows).to_csv(TABLES_DIR / 'regression_models_logistic.csv', index=False)
print('Saved: regression_models_logistic.csv')

# =============================================================================
# PART D: RACE INTERACTION TESTS (F-tests)
# =============================================================================
print('\n' + '=' * 70)
print('PART D: RACE INTERACTION TESTS')
print('=' * 70)

# Test: Officer count x Race
print('\n--- Officer Count x Race Interaction ---')
reg_df['oc_x_black'] = reg_df['num_involved_officers'] * reg_df['race_black']
reg_df['oc_x_hispanic'] = reg_df['num_involved_officers'] * reg_df['race_hispanic']

# Unrestricted model (with interactions)
m_int_vars = regions_vars + ['oc_x_black', 'oc_x_hispanic']
model_int = fit_lpm(reg_df, 'fatal', m_int_vars, agency_fe=True)  # same specification as M7

# F-test on joint significance of interaction terms
f_test = model_int.f_test('oc_x_black = 0, oc_x_hispanic = 0')
f_stat = float(f_test.fvalue)
f_p = float(f_test.pvalue)

print(f'F-test statistic: {f_stat:.2f}, df = 2, p = {f_p:.4f}')
if f_p > 0.05:
    print('Result: Officer count effect does NOT differ significantly by race')
else:
    print('Result: Officer count effect differs significantly by race')

# Race-stratified officer count coefficients
print('\n--- Officer Count Coefficient by Race (pp) ---')
interaction_results = []
for race in main_races:
    subset = reg_df[reg_df['race_std'] == race]
    try:
        m_sub = fit_lpm(subset, 'fatal', ['num_involved_officers', 'age_10yr', 'female',
                                          'armed_assault'] + wound_loc_vars)
        coef_val = m_sub.params['num_involved_officers'] * 100
        ci_low = m_sub.conf_int().loc['num_involved_officers', 0] * 100
        ci_high = m_sub.conf_int().loc['num_involved_officers', 1] * 100
        p_val = m_sub.pvalues['num_involved_officers']
        print(f'  {race}: Coef = {coef_val:.2f} pp ({ci_low:.2f}-{ci_high:.2f}), p = {p_val:.4f}')
        interaction_results.append({
            'race': race, 'n': len(subset), 'Coef': coef_val,
            'CI_lower': ci_low, 'CI_upper': ci_high, 'p_value': p_val
        })
    except Exception as e:
        print(f'  {race}: Model failed - {e}')

# =============================================================================
# PART E: SUBGROUP ANALYSES
# =============================================================================
print('\n' + '=' * 70)
print('PART E: SUBGROUP ANALYSES')
print('=' * 70)

subgroup_results = []


def run_subgroup(data, label, covariates):
    """Run OLS on a subgroup and return race + officer coefficients."""
    try:
        m = fit_lpm(data, 'fatal', covariates)
        result = {'Subgroup': label, 'N': len(data)}
        for race_var, race_label in [('race_black', 'Black'), ('race_hispanic', 'Hispanic')]:
            if race_var in covariates:
                result[f'{race_label}_Coef'] = m.params[race_var] * 100
                result[f'{race_label}_CI_lower'] = m.conf_int().loc[race_var, 0] * 100
                result[f'{race_label}_CI_upper'] = m.conf_int().loc[race_var, 1] * 100
                result[f'{race_label}_p'] = m.pvalues[race_var]
        if 'num_involved_officers' in covariates:
            result['Officer_Coef'] = m.params['num_involved_officers'] * 100
            result['Officer_CI_lower'] = m.conf_int().loc['num_involved_officers', 0] * 100
            result['Officer_CI_upper'] = m.conf_int().loc['num_involved_officers', 1] * 100
            result['Officer_p'] = m.pvalues['num_involved_officers']
        return result
    except Exception as e:
        return {'Subgroup': label, 'N': len(data), 'Note': str(e)}


# Full covariates for subgroup models
full_covs = ['race_black', 'race_hispanic', 'age_10yr', 'female', 'armed_assault',
             'num_involved_officers', 'wound_head_neck', 'wound_chest', 'wound_abdomen',
             'wound_back']

# High-risk wounds
high_risk = reg_df[reg_df['wound_location_std'].isin(['Head/Neck', 'Chest', 'Abdomen'])]
covs_hr = ['race_black', 'race_hispanic', 'age_10yr', 'female',
           'num_involved_officers', 'wound_chest', 'wound_abdomen', 'wound_back']
res = run_subgroup(high_risk, 'High-risk wounds', covs_hr)
subgroup_results.append(res)
print(f'\nHigh-risk wounds (n={res["N"]}): Black Coef = {res.get("Black_Coef", "NA")}, '
      f'Officer Coef = {res.get("Officer_Coef", "NA")}')

# Males only
males = reg_df[reg_df['gender_std'] == 'Male']
covs_males = ['race_black', 'race_hispanic', 'age_10yr', 'armed_assault',
              'num_involved_officers', 'wound_head_neck', 'wound_chest', 'wound_abdomen',
             'wound_back']
res = run_subgroup(males, 'Males only', covs_males)
subgroup_results.append(res)
print(f'Males (n={res["N"]}): Black Coef = {res.get("Black_Coef", "NA"):.2f}, '
      f'Officer Coef = {res.get("Officer_Coef", "NA"):.2f}')

# Females only
females = reg_df[reg_df['gender_std'] == 'Female']
if len(females) >= 30:
    covs_females = ['race_black', 'race_hispanic', 'age_10yr']
    res = run_subgroup(females, 'Females only', covs_females)
    subgroup_results.append(res)
    print(f'Females (n={res["N"]}): Black Coef = {res.get("Black_Coef", "NA")}')
else:
    subgroup_results.append({'Subgroup': 'Females only', 'N': len(females), 'Note': 'Too few'})
    print(f'Females (n={len(females)}): Too few for stable regression')

# Younger (<35)
younger = reg_df[reg_df['age_numeric'] < 35]
covs_age = ['race_black', 'race_hispanic', 'female', 'armed_assault',
            'num_involved_officers', 'wound_head_neck', 'wound_chest', 'wound_abdomen',
             'wound_back']
res = run_subgroup(younger, 'Younger (<35)', covs_age)
subgroup_results.append(res)
print(f'Younger (<35, n={res["N"]}): Black Coef = {res.get("Black_Coef", "NA"):.2f}')

# Older (35+)
older = reg_df[reg_df['age_numeric'] >= 35]
res = run_subgroup(older, 'Older (35+)', covs_age)
subgroup_results.append(res)
print(f'Older (35+, n={res["N"]}): Black Coef = {res.get("Black_Coef", "NA"):.2f}')

# Armed only
armed = reg_df[reg_df['armed_assault'] == 1]
covs_armed = ['race_black', 'race_hispanic', 'age_10yr', 'female',
              'num_involved_officers', 'wound_head_neck', 'wound_chest', 'wound_abdomen',
             'wound_back']
res = run_subgroup(armed, 'Armed only', covs_armed)
subgroup_results.append(res)
print(f'Armed (n={res["N"]}): Black Coef = {res.get("Black_Coef", "NA"):.2f}, '
      f'Officer Coef = {res.get("Officer_Coef", "NA"):.2f}')

# =============================================================================
# PART F: SAVE ALL RESULTS
# =============================================================================
print('\n' + '=' * 70)
print('PART F: SAVING RESULTS')
print('=' * 70)

# Save regression model results (wide format for inline code)
model_rows = []
for model_name, res in results_dict.items():
    model_rows.append(res)
pd.DataFrame(model_rows).to_csv(TABLES_DIR / 'regression_models.csv', index=False)
print('Saved: regression_models.csv')

# Save full regression table (long format)
table_rows = []
var_map = {
    'race_black': 'Black', 'race_hispanic': 'Hispanic',
    'age_10yr': 'Age (10yr)', 'female': 'Female',
    'armed_assault': 'Armed/Assault',
    'num_involved_officers': 'Officer Count',
    'wound_head_neck': 'Head/Neck', 'wound_chest': 'Chest',
    'wound_abdomen': 'Abdomen', 'wound_back': 'Back',
    'wound_count': 'Regions Wounded',
    'civ_fired_at_officer': 'Civilian Fired at Officer',
    'civ_threat_firearm': 'Firearm Threat to Officer', 'officer_injured': 'Officer Injured',
    'civ_vehicle_against_officer': 'Vehicle Used Against Officer',
    'weapon_firearm': 'Weapon: Firearm', 'weapon_replica': 'Weapon: Replica Firearm',
    'weapon_knife': 'Weapon: Knife/Blade', 'weapon_other': 'Weapon: Other',
    'contact_stop': 'Contact: Stop', 'contact_planned': 'Contact: Planned/Warrant',
    'contact_other': 'Contact: Other', 'multi_civilian': 'Multiple Civilians'
}


for model_name, (model_obj, vars_list) in models_fitted.items():
    for var in vars_list:
        label = var_map.get(var, var)
        table_rows.append({
            'Model': model_name, 'Variable': label,
            'Coef': model_obj.params[var] * 100,
            'CI_Lower': model_obj.conf_int().loc[var, 0] * 100,
            'CI_Upper': model_obj.conf_int().loc[var, 1] * 100,
            'p_value': model_obj.pvalues[var]
        })
    table_rows.append({
        'Model': model_name, 'Variable': 'R2',
        'Coef': model_obj.rsquared,
        'CI_Lower': np.nan, 'CI_Upper': np.nan, 'p_value': np.nan
    })
    table_rows.append({
        'Model': model_name, 'Variable': 'N',
        'Coef': int(model_obj.nobs),
        'CI_Lower': np.nan, 'CI_Upper': np.nan, 'p_value': np.nan
    })

pd.DataFrame(table_rows).to_csv(TABLES_DIR / 'regression_table_long.csv', index=False)
print('Saved: regression_table_long.csv')

# Save chi-square test
chi_results = {
    'chi2': chi2_wounds, 'df': dof_wounds, 'p_value': p_wounds,
    'test': 'Race x Wound Location Independence'
}
pd.DataFrame([chi_results]).to_csv(TABLES_DIR / 'chi_square_differential_aiming.csv', index=False)
print('Saved: chi_square_differential_aiming.csv')

# Save wound location x officer count test
# Test whether wound location distribution differs by single vs multi-officer
single_off = reg_df[reg_df['num_involved_officers'] == 1]
multi_off = reg_df[reg_df['num_involved_officers'] >= 2]
contingency_wound_off = pd.crosstab(reg_df['wound_location_std'],
                                     (reg_df['num_involved_officers'] >= 2).astype(int))
chi2_wo, p_wo, dof_wo, _ = stats.chi2_contingency(contingency_wound_off)

pct_head_neck_single = 100 * (single_off['wound_location_std'] == 'Head/Neck').mean()
pct_head_neck_multi = 100 * (multi_off['wound_location_std'] == 'Head/Neck').mean()
pct_lethal_single = 100 * single_off['wound_location_std'].isin(['Head/Neck', 'Chest']).mean()
pct_lethal_multi = 100 * multi_off['wound_location_std'].isin(['Head/Neck', 'Chest']).mean()

wound_off_results = {
    'chi2': chi2_wo, 'df': dof_wo, 'p_value': p_wo,
    'test': 'Wound Location x Officer Count Independence',
    'pct_head_neck_single': pct_head_neck_single,
    'pct_head_neck_multi': pct_head_neck_multi,
    'pct_lethal_single': pct_lethal_single,
    'pct_lethal_multi': pct_lethal_multi,
    'n_single': len(single_off),
    'n_multi': len(multi_off)
}
pd.DataFrame([wound_off_results]).to_csv(TABLES_DIR / 'chi_square_wound_location_officers.csv', index=False)
print('Saved: chi_square_wound_location_officers.csv')
print(f'  Chi-square: {chi2_wo:.2f}, df = {dof_wo}, p = {p_wo:.3f}')
print(f'  % head/neck+chest: single={pct_lethal_single:.1f}%, multi={pct_lethal_multi:.1f}%')

# Save race interaction test
interaction_summary = {
    'test': 'Officer Count x Race F-Test',
    'statistic': f_stat, 'df': 2, 'p_value': f_p
}
pd.DataFrame([interaction_summary]).to_csv(TABLES_DIR / 'race_interaction_test.csv', index=False)
pd.DataFrame(interaction_results).to_csv(TABLES_DIR / 'officer_coef_by_race.csv', index=False)
print('Saved: race_interaction_test.csv, officer_coef_by_race.csv')

# Save subgroup results
pd.DataFrame(subgroup_results).to_csv(TABLES_DIR / 'subgroup_analysis.csv', index=False)
print('Saved: subgroup_analysis.csv')

# Save regression data for downstream scripts
reg_df.to_csv(OUTPUT_DIR / 'california_regression_data.csv', index=False)
print(f'Saved: california_regression_data.csv ({len(reg_df):,} cases)')

# Save descriptive summary for inline code
desc_summary = {
    'n_total': n_total, 'n_fatal': n_fatal, 'cfr_overall': cfr,
    'n_male': n_male, 'n_female': n_female,
    'mean_age': mean_age, 'sd_age': std_age,
    'n_regression': len(reg_df),
    'chi2_race_wound': chi2_wounds, 'p_race_wound': p_wounds,
    'f_stat_interaction': f_stat, 'p_interaction': f_p
}
for race in main_races:
    sub = df_main[df_main['race_std'] == race]
    desc_summary[f'n_{race.lower()}'] = len(sub)
    desc_summary[f'cfr_{race.lower()}'] = sub['fatal'].mean() * 100
for loc in wound_locs:
    sub = df[df['wound_location_std'] == loc]
    key = loc.lower().replace('/', '_')
    desc_summary[f'n_{key}'] = len(sub)
    desc_summary[f'cfr_{key}'] = sub['fatal'].mean() * 100

pd.DataFrame([desc_summary]).to_csv(TABLES_DIR / 'descriptive_summary.csv', index=False)
print('Saved: descriptive_summary.csv')

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print('\n' + '=' * 70)
print('KEY RESULTS SUMMARY')
print('=' * 70)

m3_res = results_dict['Model 3']
m4_res = results_dict['Model 4']
m5_res = results_dict['Model 7']

print(f'\nRegression sample: {len(reg_df):,}')
print(f'Overall CFR: {cfr:.1f}%')

print(f'\nModel 3 (+ Resistance):')
print(f'  Officer Count Coef: {m3_res["Num Involved Officers_Coef"]:.2f} pp')
print(f'  Black Coef: {m3_res["Black_Coef"]:.2f} pp')
print(f'  Hispanic Coef: {m3_res["Hispanic_Coef"]:.2f} pp')

print(f'\nModel 4 (+ Incident Circumstances):')
print(f'  Officer Count Coef: {m4_res["Num Involved Officers_Coef"]:.2f} pp')
print(f'  Black Coef: {m4_res["Black_Coef"]:.2f} pp')
print(f'  Hispanic Coef: {m4_res["Hispanic_Coef"]:.2f} pp')

print(f'\nModel 7 (+ Agency FE, Wound Location, Regions Wounded):')
print(f'  Officer Count Coef: {m5_res["Num Involved Officers_Coef"]:.2f} pp')
print(f'  Black Coef: {m5_res["Black_Coef"]:.2f} pp')
print(f'  Hispanic Coef: {m5_res["Hispanic_Coef"]:.2f} pp')

print(f'\nDifferential aiming: chi2 = {chi2_wounds:.2f}, p = {p_wounds:.3f}')
print(f'Race interaction (F-test): F = {f_stat:.2f}, p = {f_p:.4f}')

print('\n' + '=' * 70)
print('ANALYSIS COMPLETE')
print('=' * 70)

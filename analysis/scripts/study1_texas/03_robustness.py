"""
Study 1 - Texas: Robustness checks for officer count analysis.

Multiple sensitivity analyses to assess robustness of main finding.
Primary specification: OLS (linear probability model) with HC1 robust SEs.

Adapted from officer_count_paper/Study 1 - Texas/scripts/04_robustness_checks.py
for the merged manuscript (Nix & Adams).
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
OUTPUT_DIR = MERGED_ROOT / 'outputs' / 'study1'
TABLES_DIR = OUTPUT_DIR / 'tables'
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 70)
print('MERGED PAPER - STUDY 1: ROBUSTNESS CHECKS')
print('Texas Justice Initiative OIS Data')
print('Primary specification: OLS (linear probability model)')
print('=' * 70)

# =============================================================================
# LOAD DATA
# =============================================================================
# Regression sample with all M4 variables, year, and agency (from 02_run_analysis.py).
# Revised 2026-10-01: every specification uses the primary estimator (year FE, agency-clustered
# SEs) unless the specification itself varies the estimator.
reg_df = pd.read_csv(OUTPUT_DIR / 'texas_regression_data.csv')
reg_df['log_officers'] = np.log(reg_df['Officer_Count'])
CIRC = ['officer_shot_by_civ', 'weapon_firearm', 'weapon_knife', 'weapon_vehicle', 'weapon_other',
        'call_traffic', 'call_warrant', 'call_other', 'multi_civilian']
FULL = ['age_10yr', 'female', 'race_black', 'race_hispanic', 'armed'] + CIRC
print(f'Base regression sample: {len(reg_df):,} cases')

robustness_results = []


def run_model(data, officer_var='Officer_Count', covariates=None, label='', **fit_kwargs):
    """Run the LPM (primary estimator unless fit_kwargs override) and return key results."""
    if covariates is None:
        covariates = FULL
    try:
        model = fit_lpm(data, 'fatal', [officer_var] + covariates, **fit_kwargs)
        ci = model.conf_int().loc[officer_var] * 100
        p_val = model.pvalues[officer_var]
        return {'specification': label, 'n': int(model.nobs), 'Coef': model.params[officer_var] * 100,
                'CI_lower': ci[0], 'CI_upper': ci[1], 'p_value': p_val, 'significant': p_val < 0.05}
    except Exception as e:
        return {'specification': label, 'n': len(data), 'Coef': np.nan, 'CI_lower': np.nan,
                'CI_upper': np.nan, 'p_value': np.nan, 'significant': False, 'error': str(e)}


# =============================================================================
# BASELINE
# =============================================================================
result = run_model(reg_df, label='1. Baseline (M4: full covariates)')
robustness_results.append(result)
print(f"\nBaseline: Coef = {result['Coef']:.2f} pp ({result['CI_lower']:.2f}-{result['CI_upper']:.2f}), p = {result['p_value']:.4f}")

# =============================================================================
# ALTERNATIVE FUNCTIONAL FORMS
# =============================================================================
print('\n--- Alternative Functional Forms ---')

result = run_model(reg_df, officer_var='log_officers', label='2. Log(officer count)')
robustness_results.append(result)
print(f"Log officers: Coef = {result['Coef']:.2f} pp ({result['CI_lower']:.2f}-{result['CI_upper']:.2f})")

reg_df['officer_capped5'] = reg_df['Officer_Count'].clip(upper=5)
result = run_model(reg_df, officer_var='officer_capped5', label='3. Capped at 5 officers')
robustness_results.append(result)
print(f"Capped at 5: Coef = {result['Coef']:.2f} pp ({result['CI_lower']:.2f}-{result['CI_upper']:.2f})")

reg_df['multiple'] = (reg_df['Officer_Count'] >= 2).astype(int)
result = run_model(reg_df, officer_var='multiple', label='4. Binary (1 vs 2+)')
robustness_results.append(result)
print(f"Binary 2+: Coef = {result['Coef']:.2f} pp ({result['CI_lower']:.2f}-{result['CI_upper']:.2f})")

reg_df['three_plus'] = (reg_df['Officer_Count'] >= 3).astype(int)
result = run_model(reg_df, officer_var='three_plus', label='5. Binary (1-2 vs 3+)')
robustness_results.append(result)
print(f"Binary 3+: Coef = {result['Coef']:.2f} pp ({result['CI_lower']:.2f}-{result['CI_upper']:.2f})")

# =============================================================================
# DIFFERENT COVARIATE SETS
# =============================================================================
print('\n--- Different Covariate Sets ---')

result = run_model(reg_df, covariates=[], label='6. Unadjusted')
robustness_results.append(result)
print(f"Unadjusted: Coef = {result['Coef']:.2f} pp")

result = run_model(reg_df, covariates=['age_10yr', 'female', 'race_black', 'race_hispanic'],
                   label='7. Demographics only')
robustness_results.append(result)
print(f"Demo only: Coef = {result['Coef']:.2f} pp")

# =============================================================================
# SAMPLE RESTRICTIONS
# =============================================================================
print('\n--- Sample Restrictions ---')

males_df = reg_df[reg_df['female'] == 0]
result = run_model(males_df, covariates=[c for c in FULL if c != 'female'],
                   label='8. Males only')
robustness_results.append(result)
print(f"Males only: Coef = {result['Coef']:.2f} pp (n={result['n']})")

armed_df = reg_df[reg_df['armed'] == 1]
result = run_model(armed_df, covariates=[c for c in FULL if c != 'armed'],
                   label='9. Armed civilians only')
robustness_results.append(result)
print(f"Armed only: Coef = {result['Coef']:.2f} pp (n={result['n']})")

no_outliers = reg_df[reg_df['Officer_Count'] <= 10]
result = run_model(no_outliers, label='10. Exclude officer count > 10')
robustness_results.append(result)
print(f"No outliers: Coef = {result['Coef']:.2f} pp (n={result['n']})")

adults = reg_df[reg_df['age_numeric'] >= 18]
result = run_model(adults, label='11. Adults only (18+)')
robustness_results.append(result)
print(f"Adults only: Coef = {result['Coef']:.2f} pp (n={result['n']})")

# =============================================================================
# TIME PERIOD RESTRICTIONS
# =============================================================================
print('\n--- Time Period Restrictions ---')

recent = reg_df[reg_df['incident_year'] >= 2020]
if len(recent) >= 100:
    result = run_model(recent, label='12. 2020 onwards')
    robustness_results.append(result)
    print(f"2020+: Coef = {result['Coef']:.2f} pp (n={result['n']})")

# =============================================================================
# THREAT-LEVEL SUBGROUP (UNARMED) — added in R1 revision (Phase 2 blindspot)
# =============================================================================
print('\n--- Threat-Level Subgroup ---')

unarmed_df = reg_df[reg_df['armed'] == 0]
result = run_model(unarmed_df,
                   covariates=[c for c in FULL if c != 'armed'],
                   label='13. Unarmed civilians only')
robustness_results.append(result)
print(f"Unarmed only: Coef = {result['Coef']:.2f} pp (n={result['n']})")

# =============================================================================
# ESTIMATOR AND CONTROL VARIANTS (added 2026-10-01, PQ R&R)
# =============================================================================
print('\n--- Estimator Variants ---')
for label, kwargs, covs in [
        ('14. M3 covariates (no incident circumstances)', {}, ['age_10yr', 'female', 'race_black', 'race_hispanic', 'armed']),
        ('15. HC1 SEs, no year FE (Round 1 estimator)', {'year_fe': False, 'cov': 'HC1'}, None),
        ('16. HC3 SEs', {'cov': 'HC3'}, None),
        ('17. + Agency fixed effects', {'agency_fe': True}, None)]:
    result = run_model(reg_df, covariates=covs, label=label, **kwargs)
    robustness_results.append(result)
    print(f"{label}: Coef = {result['Coef']:.2f} pp, p = {result['p_value']:.4f}")

# =============================================================================
# DATA-SOURCE CHECKS (added 2026-10-02, study period extended through 2025)
# =============================================================================
# Cases after 2024-08-26, and OAG-reported cases missing from TJI, come from OAG reports and
# have no TJI media-weapon coding (coded Missing); officer counts were corrected against the
# OAG reports for multi-agency shootings. These specifications isolate each change.
print('\n--- Data-Source Checks ---')
no_media = [c for c in FULL if not c.startswith('weapon_')]
result = run_model(reg_df, covariates=no_media, label='18. Without media-reported weapon controls')
robustness_results.append(result)
print(f"No media weapon: Coef = {result['Coef']:.2f} pp, p = {result['p_value']:.4f}")

tji_era = reg_df[reg_df['source'] == 'TJI']
result = run_model(tji_era, label='19. TJI records only (no OAG-only cases)')
robustness_results.append(result)
print(f"TJI era only: Coef = {result['Coef']:.2f} pp (n={result['n']})")

reg_df['tji_count'] = reg_df['tji_officer_count'].fillna(reg_df['Officer_Count'])
result = run_model(reg_df, officer_var='tji_count', label="20. TJI's original officer counts")
robustness_results.append(result)
print(f"TJI counts: Coef = {result['Coef']:.2f} pp, p = {result['p_value']:.4f}")

no_gap = reg_df[reg_df['source'] != 'OAG (missing from TJI)']
result = run_model(no_gap, label='21. Excluding OAG cases missing from TJI')
robustness_results.append(result)
print(f"No TJI-gap cases: Coef = {result['Coef']:.2f} pp (n={result['n']})")

# =============================================================================
# SUMMARY
# =============================================================================
print('\n' + '=' * 70)
print('ROBUSTNESS SUMMARY')
print('=' * 70)

robust_df = pd.DataFrame(robustness_results)
n_sig = robust_df['significant'].sum()
n_total = len(robust_df)
n_positive = (robust_df['Coef'] > 0).sum()

print(f'\nTotal specifications: {n_total}')
print(f'Specifications with Coef > 0: {n_positive} ({n_positive/n_total*100:.0f}%)')
print(f'Statistically significant (p < 0.05): {n_sig} ({n_sig/n_total*100:.0f}%)')

print(f'\n{"Specification":<35} {"N":<8} {"Coef(pp)":<10} {"95% CI":<20} {"p":<10}')
print('-' * 85)
for _, row in robust_df.iterrows():
    if pd.notna(row['Coef']):
        spec = row['specification'][:35]
        ci_str = f"({row['CI_lower']:.2f}-{row['CI_upper']:.2f})"
        sig = '*' if row['significant'] else ''
        print(f'{spec:<35} {int(row["n"]):<8} {row["Coef"]:.2f}     {ci_str:<20} {row["p_value"]:.4f}   {sig}')

# Save
robust_df.to_csv(TABLES_DIR / 'robustness_results.csv', index=False)
print(f'\nSaved: robustness_results.csv ({len(robust_df)} specifications)')

print('\n' + '=' * 70)
print('ROBUSTNESS CHECKS COMPLETE')
print('=' * 70)

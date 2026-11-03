"""
REVISION: Nonlinear officer count specifications.

Tests whether the officer count-fatality association is truly linear
or whether a binary threshold (2+ officers) or quadratic fits better.

Addresses Reviewer 2's concern that the linear specification is an
assumption treated as a finding.

Outputs:
  outputs/revision/nonlinear_officer_models_tx.csv
  outputs/revision/nonlinear_officer_models_ca.csv
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
OUTPUT_DIR = MERGED_ROOT / 'outputs' / 'revision'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 70)
print('REVISION: NONLINEAR OFFICER COUNT TESTS')
print('=' * 70)


def fit_and_extract(df, xvars, label):
    """Fit the primary specification (year FE, agency-clustered SEs) and return fit stats.

    AIC/BIC come from the OLS log-likelihood, which the SE choice does not affect.
    """
    model = fit_lpm(df, 'fatal', xvars)
    return {
        'specification': label,
        'R2': model.rsquared,
        'AIC': model.aic,
        'BIC': model.bic,
        'N': int(model.nobs),
    }, model


def run_nonlinear_tests(df, officer_col, covariates, state_name):
    """Run all nonlinear specifications for one state."""
    results = []

    # --- Spec 1: Linear (baseline) ---
    res_lin, model_lin = fit_and_extract(df, [officer_col] + covariates, 'Linear (baseline)')
    res_lin['officer_coef_pp'] = model_lin.params[officer_col] * 100
    res_lin['officer_ci_lower'] = model_lin.conf_int().loc[officer_col, 0] * 100
    res_lin['officer_ci_upper'] = model_lin.conf_int().loc[officer_col, 1] * 100
    res_lin['officer_p'] = model_lin.pvalues[officer_col]
    results.append(res_lin)

    # --- Spec 2: Quadratic ---
    sq_col = f'{officer_col}_sq'
    df[sq_col] = df[officer_col] ** 2
    res_quad, model_quad = fit_and_extract(df, [officer_col, sq_col] + covariates, 'Quadratic')
    res_quad['officer_coef_pp'] = model_quad.params[officer_col] * 100
    res_quad['officer_ci_lower'] = model_quad.conf_int().loc[officer_col, 0] * 100
    res_quad['officer_ci_upper'] = model_quad.conf_int().loc[officer_col, 1] * 100
    res_quad['officer_p'] = model_quad.pvalues[officer_col]
    res_quad['quadratic_coef'] = model_quad.params[sq_col] * 100
    res_quad['quadratic_p'] = model_quad.pvalues[sq_col]
    results.append(res_quad)

    # F-test: quadratic vs linear (is the quadratic term significant?)
    # This is just the t-test on the squared term in the quadratic model
    f_stat_quad = (model_quad.params[sq_col] / model_quad.bse[sq_col]) ** 2
    f_p_quad = model_quad.pvalues[sq_col]

    # --- Spec 3: Binary 2+ threshold ---
    bin_col = 'officer_2plus'
    df[bin_col] = (df[officer_col] >= 2).astype(int)
    res_bin, model_bin = fit_and_extract(df, [bin_col] + covariates, 'Binary (2+ vs 1)')
    res_bin['officer_coef_pp'] = model_bin.params[bin_col] * 100
    res_bin['officer_ci_lower'] = model_bin.conf_int().loc[bin_col, 0] * 100
    res_bin['officer_ci_upper'] = model_bin.conf_int().loc[bin_col, 1] * 100
    res_bin['officer_p'] = model_bin.pvalues[bin_col]
    results.append(res_bin)

    # --- Spec 4: Step-at-2-then-linear (allows jump at 2, then linear beyond) ---
    # Coded as: binary_2plus + max(officer_count - 2, 0): a discrete jump at 2 and a separate
    # per-officer slope beyond 2. (Using officer_count - 1 instead would fold one slope unit
    # into the jump.)
    # officer_above1 = officers beyond the second (0 for 1 or 2 officers)
    df['officer_above1'] = np.maximum(df[officer_col] - 2, 0)
    res_step, model_step = fit_and_extract(df, [bin_col, 'officer_above1'] + covariates,
                                           'Step at 2 + slope beyond')
    res_step['officer_coef_pp'] = model_step.params[bin_col] * 100  # jump at 2
    res_step['officer_ci_lower'] = model_step.conf_int().loc[bin_col, 0] * 100
    res_step['officer_ci_upper'] = model_step.conf_int().loc[bin_col, 1] * 100
    res_step['officer_p'] = model_step.pvalues[bin_col]
    res_step['slope_beyond2_pp'] = model_step.params['officer_above1'] * 100
    res_step['slope_beyond2_p'] = model_step.pvalues['officer_above1']
    res_step['slope_beyond2_ci_lower'] = model_step.conf_int().loc['officer_above1', 0] * 100
    res_step['slope_beyond2_ci_upper'] = model_step.conf_int().loc['officer_above1', 1] * 100
    results.append(res_step)

    # --- Spec 5: Categorical dummies (most flexible) ---
    df['off_2'] = (df[officer_col] == 2).astype(int)
    df['off_3'] = (df[officer_col] == 3).astype(int)
    df['off_4plus'] = (df[officer_col] >= 4).astype(int)
    res_cat, model_cat = fit_and_extract(df, ['off_2', 'off_3', 'off_4plus'] + covariates,
                                         'Categorical dummies')
    res_cat['officer_coef_pp'] = model_cat.params['off_2'] * 100  # 2 vs 1
    res_cat['off_3_pp'] = model_cat.params['off_3'] * 100
    res_cat['off_4plus_pp'] = model_cat.params['off_4plus'] * 100
    # Joint F-test on all dummies
    # Wald chi-square (cluster-robust); the F form is undefined with many singleton clusters
    w = model_cat.wald_test('off_2 = 0, off_3 = 0, off_4plus = 0', scalar=True, use_f=False)
    res_cat['joint_F'] = float(w.statistic) / 3
    res_cat['joint_F_p'] = float(w.pvalue)
    for d in ['off_2', 'off_3', 'off_4plus']:
        res_cat[f'{d}_ci_lower'] = model_cat.conf_int().loc[d, 0] * 100
        res_cat[f'{d}_ci_upper'] = model_cat.conf_int().loc[d, 1] * 100
    results.append(res_cat)

    results_df = pd.DataFrame(results)

    # Print summary
    print(f'\n--- {state_name} ---')
    print(f'{"Specification":<30} {"Coef (pp)":<12} {"p":<10} {"AIC":<12} {"BIC":<12}')
    print('-' * 76)
    for _, row in results_df.iterrows():
        print(f'{row["specification"]:<30} {row["officer_coef_pp"]:>8.2f}   '
              f'{row["officer_p"]:>8.4f}   {row["AIC"]:>10.1f}   {row["BIC"]:>10.1f}')

    print(f'\nQuadratic term p-value: {f_p_quad:.4f}')
    print(f'Interpretation: {"Nonlinear" if f_p_quad < 0.05 else "Linear specification adequate"}')

    if 'slope_beyond2_pp' in results_df.columns:
        step_row = results_df[results_df['specification'] == 'Step at 2 + slope beyond'].iloc[0]
        print(f'Step model: jump at 2 = {step_row["officer_coef_pp"]:.2f} pp, '
              f'slope beyond 2 = {step_row["slope_beyond2_pp"]:.2f} pp (p = {step_row["slope_beyond2_p"]:.4f})')

    return results_df


# =============================================================================
# TEXAS
# =============================================================================
print('\n' + '=' * 70)
print('TEXAS')
print('=' * 70)

tx = pd.read_csv(MERGED_ROOT / 'outputs' / 'study1' / 'texas_regression_data.csv')

tx_covariates = ['age_10yr', 'female', 'race_black', 'race_hispanic', 'armed']
tx_results = run_nonlinear_tests(tx, 'Officer_Count', tx_covariates, 'Texas')
tx_results.to_csv(OUTPUT_DIR / 'nonlinear_officer_models_tx.csv', index=False)
print(f'\nSaved: nonlinear_officer_models_tx.csv')

# =============================================================================
# CALIFORNIA
# =============================================================================
print('\n' + '=' * 70)
print('CALIFORNIA')
print('=' * 70)

# Same covariates as TX (M3): total associations, not adjusted for post-exposure wound
# variables (no wound-location terms in either state).
ca = pd.read_csv(MERGED_ROOT / 'outputs' / 'study2' / 'california_regression_data.csv')
ca_covariates = ['age_10yr', 'female', 'race_black', 'race_hispanic', 'armed_assault']
ca_results = run_nonlinear_tests(ca, 'num_involved_officers', ca_covariates, 'California')
ca_results.to_csv(OUTPUT_DIR / 'nonlinear_officer_models_ca.csv', index=False)
print(f'\nSaved: nonlinear_officer_models_ca.csv')

print('\n' + '=' * 70)
print('NONLINEAR TESTS COMPLETE')
print('=' * 70)

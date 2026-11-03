"""
REVISION: OLS vs Logistic average marginal effect comparison.

Demonstrates that OLS coefficients and logistic AMEs yield substantively
identical conclusions, addressing Reviewer 3's concern that logistic
models should be more central.

Outputs:
  outputs/revision/logistic_ols_comparison.csv
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
MERGED_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = MERGED_ROOT / 'outputs' / 'revision'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 70)
print('REVISION: OLS vs LOGISTIC AME COMPARISON')
print('=' * 70)


def compute_ame(logit_model, X_data):
    """Compute average marginal effects from a logistic model."""
    # Predicted probabilities
    p_hat = logit_model.predict(X_data)
    # For logit: AME = mean(p*(1-p)) * beta
    density = np.mean(p_hat * (1 - p_hat))

    ames = {}
    for var in logit_model.params.index:
        if var == 'const':
            continue
        ames[var] = density * logit_model.params[var]
    return ames


results = []

# =============================================================================
# TEXAS MODELS (M1-M3)
# =============================================================================
print('\n--- Texas ---')
tx = pd.read_csv(MERGED_ROOT / 'outputs' / 'study1' / 'texas_analysis_ready.csv')
tx = tx[tx['Officer_Count'].notna()].copy()
tx = tx[tx['race_std'].isin(['White', 'Black', 'Hispanic'])]
tx = tx.dropna(subset=['age_numeric', 'Victim_Gender', 'Officer_Count'])
tx = tx[tx['Victim_Gender'].isin(['MALE', 'FEMALE', 'Male', 'Female'])]

tx['race_black'] = (tx['race_std'] == 'Black').astype(int)
tx['race_hispanic'] = (tx['race_std'] == 'Hispanic').astype(int)
tx['female'] = tx['Victim_Gender'].str.upper().eq('FEMALE').astype(int)
tx['age_10yr'] = tx['age_numeric'] / 10
tx['armed'] = tx['had_weapon'].astype(int)

y_tx = tx['fatal']

tx_specs = {
    'M1: Officer Count': ['Officer_Count'],
    'M2: + Demographics': ['Officer_Count', 'age_10yr', 'female', 'race_black', 'race_hispanic'],
    'M3: Full': ['Officer_Count', 'age_10yr', 'female', 'race_black', 'race_hispanic', 'armed'],
}

for model_name, vars_list in tx_specs.items():
    X = sm.add_constant(tx[vars_list])
    ols = sm.OLS(y_tx, X).fit(cov_type='HC1')
    logit = sm.Logit(y_tx, X).fit(disp=0)
    ames = compute_ame(logit, X)

    for var in vars_list:
        ols_coef = ols.params[var] * 100
        ols_p = ols.pvalues[var]
        ame = ames[var] * 100
        logit_p = logit.pvalues[var]

        results.append({
            'state': 'Texas',
            'model': model_name,
            'variable': var,
            'ols_coef_pp': ols_coef,
            'ols_p': ols_p,
            'logit_ame_pp': ame,
            'logit_p': logit_p,
            'difference_pp': abs(ols_coef - ame),
            'same_sign': np.sign(ols_coef) == np.sign(ame),
            'same_significance': (ols_p < 0.05) == (logit_p < 0.05),
        })

# =============================================================================
# CALIFORNIA MODELS (M1-M6)
# =============================================================================
print('\n--- California ---')
ca = pd.read_csv(MERGED_ROOT / 'outputs' / 'study2' / 'california_analysis_ready.csv')
ca = ca[ca['num_involved_officers'].notna()].copy()
ca = ca[ca['race_std'].isin(['White', 'Black', 'Hispanic'])]
ca = ca.dropna(subset=['age_numeric', 'gender_std', 'num_involved_officers'])
ca = ca[ca['gender_std'] != 'Unknown']
ca = ca[ca['wound_location_std'].isin(['Head/Neck', 'Chest', 'Abdomen', 'Extremities'])]

ca['race_black'] = (ca['race_std'] == 'Black').astype(int)
ca['race_hispanic'] = (ca['race_std'] == 'Hispanic').astype(int)
ca['female'] = (ca['gender_std'] == 'Female').astype(int)
ca['age_10yr'] = ca['age_numeric'] / 10
ca['armed_assault'] = ca['resistance_std'].isin(['Armed/Deadly Force', 'Physical Assault']).astype(int)
ca['wl_head'] = (ca['wound_location_std'] == 'Head/Neck').astype(int)
ca['wl_chest'] = (ca['wound_location_std'] == 'Chest').astype(int)
ca['wl_abdomen'] = (ca['wound_location_std'] == 'Abdomen').astype(int)

y_ca = ca['fatal']

ca_specs = {
    'M1: Race': ['race_black', 'race_hispanic'],
    'M2: + Demographics': ['race_black', 'race_hispanic', 'age_10yr', 'female'],
    'M3: + Resistance': ['race_black', 'race_hispanic', 'age_10yr', 'female', 'armed_assault'],
    'M4: + Officer Count': ['race_black', 'race_hispanic', 'age_10yr', 'female',
                            'armed_assault', 'num_involved_officers'],
    'M5: + Wound Location': ['race_black', 'race_hispanic', 'age_10yr', 'female',
                             'armed_assault', 'num_involved_officers',
                             'wl_head', 'wl_chest', 'wl_abdomen'],
    'M6: + Wound Count': ['race_black', 'race_hispanic', 'age_10yr', 'female',
                          'armed_assault', 'num_involved_officers',
                          'wl_head', 'wl_chest', 'wl_abdomen', 'wound_count'],
}

for model_name, vars_list in ca_specs.items():
    X = sm.add_constant(ca[vars_list])
    ols = sm.OLS(y_ca, X).fit(cov_type='HC1')
    logit = sm.Logit(y_ca, X).fit(disp=0)
    ames = compute_ame(logit, X)

    # Focus on key variables only
    key_vars = ['num_involved_officers', 'race_black'] if 'num_involved_officers' in vars_list else ['race_black']
    for var in key_vars:
        if var not in vars_list:
            continue
        ols_coef = ols.params[var] * 100
        ols_p = ols.pvalues[var]
        ame = ames[var] * 100
        logit_p = logit.pvalues[var]

        results.append({
            'state': 'California',
            'model': model_name,
            'variable': var,
            'ols_coef_pp': ols_coef,
            'ols_p': ols_p,
            'logit_ame_pp': ame,
            'logit_p': logit_p,
            'difference_pp': abs(ols_coef - ame),
            'same_sign': np.sign(ols_coef) == np.sign(ame),
            'same_significance': (ols_p < 0.05) == (logit_p < 0.05),
        })

# =============================================================================
# SUMMARY
# =============================================================================
results_df = pd.DataFrame(results)

print('\n' + '=' * 70)
print('COMPARISON SUMMARY')
print('=' * 70)
print(f'\n{"State":<12} {"Model":<25} {"Variable":<25} {"OLS (pp)":<10} {"AME (pp)":<10} {"Diff":<8} {"Sign":<6} {"Sig":<6}')
print('-' * 102)
for _, row in results_df.iterrows():
    print(f'{row["state"]:<12} {row["model"]:<25} {row["variable"]:<25} '
          f'{row["ols_coef_pp"]:>7.2f}   {row["logit_ame_pp"]:>7.2f}   '
          f'{row["difference_pp"]:>5.2f}   {"Yes" if row["same_sign"] else "NO":>4}   '
          f'{"Yes" if row["same_significance"] else "NO":>4}')

n_same_sign = results_df['same_sign'].sum()
n_same_sig = results_df['same_significance'].sum()
n_total = len(results_df)
mean_diff = results_df['difference_pp'].mean()
max_diff = results_df['difference_pp'].max()

print(f'\nSame sign: {n_same_sign}/{n_total} ({n_same_sign/n_total*100:.0f}%)')
print(f'Same significance: {n_same_sig}/{n_total} ({n_same_sig/n_total*100:.0f}%)')
print(f'Mean absolute difference: {mean_diff:.2f} pp')
print(f'Max absolute difference: {max_diff:.2f} pp')

results_df.to_csv(OUTPUT_DIR / 'logistic_ols_comparison.csv', index=False)
print(f'\nSaved: logistic_ols_comparison.csv')

print('\n' + '=' * 70)
print('OLS vs LOGISTIC AME COMPARISON COMPLETE')
print('=' * 70)

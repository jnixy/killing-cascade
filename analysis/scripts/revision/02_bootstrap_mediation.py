"""
REVISION: Bootstrap confidence intervals for mediation analysis.

Agency-cluster nonparametric bootstrap (B=5,000), bias-corrected (BC, not BCa) intervals, of the Baron & Kenny indirect effect
(officers -> wound count -> fatality) to address Reviewer 2 and
Reviewer 3's request for proper inference on mediation pathways.

Outputs:
  outputs/revision/bootstrap_mediation_results.csv
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
from lpm_utils import fe_dummies  # noqa: E402
MERGED_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = MERGED_ROOT / 'outputs' / 'revision'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

B = 5000  # Number of bootstrap replications
SEED = 20260331

print('=' * 70)
print('REVISION: BOOTSTRAP MEDIATION CONFIDENCE INTERVALS')
print(f'B = {B:,} replications')
print('=' * 70)

# =============================================================================
# LOAD AND PREPARE DATA (same sample as 03_mediation_analysis.py)
# =============================================================================
# Same sample and covariates as 02_run_analysis.py M4 / 03_mediation_analysis.py (revised
# 2026-10-01). Year FE enter every path; resampling is by agency (cluster bootstrap) to match
# the agency-clustered primary specification.
df = pd.read_csv(MERGED_ROOT / 'outputs' / 'study2' / 'california_regression_data.csv')

print(f'Mediation sample: {len(df):,} cases')

covariates = ['age_10yr', 'female', 'race_black', 'race_hispanic', 'armed_assault',
              'civ_fired_at_officer', 'civ_threat_firearm', 'officer_injured', 'civ_vehicle_against_officer',
              'weapon_firearm', 'weapon_replica', 'weapon_knife', 'weapon_other',
              'contact_stop', 'contact_planned', 'contact_other', 'multi_civilian']
X_var = 'num_involved_officers'
M_var = 'wound_count'
Y_var = 'fatal'


def within(df, cid):
    """Demean every column within cluster id: absorbs agency fixed effects (FWL)."""
    return df - df.groupby(cid.values).transform('mean')


def mediation_paths(data):
    """Compute Baron & Kenny mediation paths for one (re)sample.

    Every path includes year FE and agency FE (revised 2026-10-02, matching Table 4 M5 and the
    regions-wounded path). Agency FE are absorbed by demeaning within '_cid', the agency id
    (unique per resampled copy of an agency in the bootstrap), which is identical to
    including agency dummies.
    """
    cid = data['_cid']
    yr = fe_dummies(data, 'year', 'yr')
    base = pd.concat([data[[X_var] + covariates], yr], axis=1).astype(float)
    y = within(data[[Y_var]].astype(float), cid)[Y_var]
    m = within(data[[M_var]].astype(float), cid)[M_var]
    X_tot = within(base, cid)
    X_b = within(pd.concat([data[[M_var]].astype(float), base], axis=1), cid)

    # Total effect: X -> Y
    total = sm.OLS(y, X_tot).fit()
    total_effect = total.params[X_var]

    # a-path: X -> M
    a_model = sm.OLS(m, X_tot).fit()
    a_path = a_model.params[X_var]

    # b-path and direct effect: M -> Y | X
    b_model = sm.OLS(y, X_b).fit()
    b_path = b_model.params[M_var]
    direct_effect = b_model.params[X_var]

    # Indirect = a * b (product method)
    indirect_ab = a_path * b_path
    # Also compute as total - direct (difference method; should match in OLS)
    indirect_diff = total_effect - direct_effect

    prop_mediated = indirect_ab / total_effect if total_effect != 0 else np.nan

    return {
        'total_effect': total_effect,
        'direct_effect': direct_effect,
        'indirect_ab': indirect_ab,
        'indirect_diff': indirect_diff,
        'a_path': a_path,
        'b_path': b_path,
        'prop_mediated': prop_mediated,
    }


# =============================================================================
# POINT ESTIMATES (original sample)
# =============================================================================
print('\nComputing point estimates...')
df['_cid'] = df['agency']
point = mediation_paths(df)

print(f'  Total effect:    {point["total_effect"]*100:.2f} pp')
print(f'  Direct effect:   {point["direct_effect"]*100:.2f} pp')
print(f'  Indirect (a*b):  {point["indirect_ab"]*100:.2f} pp')
print(f'  Indirect (diff): {point["indirect_diff"]*100:.2f} pp')
print(f'  Proportion mediated: {point["prop_mediated"]*100:.1f}%')

# =============================================================================
# BOOTSTRAP
# =============================================================================
print(f'\nRunning {B:,} bootstrap replications...')
rng = np.random.default_rng(SEED)
agencies = df['agency'].unique()
by_agency = {g: d for g, d in df.groupby('agency')}

boot_results = []
for i in range(B):
    if (i + 1) % 1000 == 0:
        print(f'  Replication {i+1:,}/{B:,}')
    draw = rng.choice(agencies, size=len(agencies), replace=True)
    # Each drawn agency gets its own id, so a twice-drawn agency gets two fixed effects
    boot_data = pd.concat([by_agency[g].assign(_cid=j) for j, g in enumerate(draw)],
                          ignore_index=True)
    try:
        boot_results.append(mediation_paths(boot_data))
    except Exception:
        continue  # skip failed replications (rare singular matrices)

boot_df = pd.DataFrame(boot_results)
n_success = len(boot_df)
print(f'\nSuccessful replications: {n_success:,}/{B:,}')

# =============================================================================
# CONFIDENCE INTERVALS
# =============================================================================
print('\n' + '=' * 70)
print('BOOTSTRAP RESULTS')
print('=' * 70)

alpha = 0.05
summary_rows = []

for param in ['total_effect', 'direct_effect', 'indirect_ab', 'a_path', 'b_path', 'prop_mediated']:
    point_est = point[param]
    boot_vals = boot_df[param].dropna().values

    # Percentile CI
    pct_lo = np.percentile(boot_vals, alpha / 2 * 100)
    pct_hi = np.percentile(boot_vals, (1 - alpha / 2) * 100)

    # Bias-corrected (BC) CI — uses bootstrap distribution only, no jackknife
    z0 = stats.norm.ppf(np.mean(boot_vals < point_est))
    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)

    bc_lo_pct = max(0.5, min(99.5, stats.norm.cdf(2 * z0 + z_lo) * 100))
    bc_hi_pct = max(0.5, min(99.5, stats.norm.cdf(2 * z0 + z_hi) * 100))

    bc_lo = np.percentile(boot_vals, bc_lo_pct)
    bc_hi = np.percentile(boot_vals, bc_hi_pct)

    boot_se = np.std(boot_vals, ddof=1)

    if param in ['total_effect', 'direct_effect', 'indirect_ab', 'a_path', 'b_path']:
        display_point = point_est * 100
        display_pct_lo = pct_lo * 100
        display_pct_hi = pct_hi * 100
        display_bc_lo = bc_lo * 100
        display_bc_hi = bc_hi * 100
        display_se = boot_se * 100
        unit = 'pp'
    else:  # prop_mediated
        display_point = point_est * 100
        display_pct_lo = pct_lo * 100
        display_pct_hi = pct_hi * 100
        display_bc_lo = bc_lo * 100
        display_bc_hi = bc_hi * 100
        display_se = boot_se * 100
        unit = '%'

    summary_rows.append({
        'parameter': param,
        'point_estimate': display_point,
        'boot_se': display_se,
        'percentile_ci_lower': display_pct_lo,
        'percentile_ci_upper': display_pct_hi,
        'bc_ci_lower': display_bc_lo,
        'bc_ci_upper': display_bc_hi,
        'unit': unit,
    })

    print(f'\n{param}:')
    print(f'  Point estimate: {display_point:.2f} {unit}')
    print(f'  Boot SE: {display_se:.2f}')
    print(f'  Percentile CI: ({display_pct_lo:.2f}, {display_pct_hi:.2f})')
    print(f'  BC CI:         ({display_bc_lo:.2f}, {display_bc_hi:.2f})')

# Key test: does the indirect effect CI exclude zero?
indirect_row = [r for r in summary_rows if r['parameter'] == 'indirect_ab'][0]
print(f'\n*** KEY RESULT: Indirect effect (a*b) ***')
print(f'  {indirect_row["point_estimate"]:.2f} pp, '
      f'BC 95% CI: ({indirect_row["bc_ci_lower"]:.2f}, {indirect_row["bc_ci_upper"]:.2f})')
if indirect_row['bc_ci_lower'] > 0:
    print(f'  CI excludes zero -> significant mediation')
else:
    print(f'  CI includes zero -> mediation not significant at 5% level')

# =============================================================================
# SAVE
# =============================================================================
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / 'bootstrap_mediation_results.csv', index=False)
print(f'\nSaved: bootstrap_mediation_results.csv')

# Also save full bootstrap distribution for diagnostics
boot_df.to_csv(OUTPUT_DIR / 'bootstrap_mediation_draws.csv', index=False)
print(f'Saved: bootstrap_mediation_draws.csv ({n_success:,} draws)')

print('\n' + '=' * 70)
print('BOOTSTRAP MEDIATION COMPLETE')
print('=' * 70)

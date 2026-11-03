"""
REVISION: Race x wound location cross-tabulation.

Reviewers 2 and 3 want a full contingency table, not just the chi-square
summary statistic. Shows counts and percentages by race and wound location.

Outputs:
  outputs/revision/race_wound_crosstab.csv
  outputs/revision/race_wound_crosstab_with_cfr.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
MERGED_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = MERGED_ROOT / 'outputs' / 'revision'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 70)
print('REVISION: RACE x WOUND LOCATION CROSS-TABULATION')
print('=' * 70)

# Load CA data
# Regression sample, so Table A5 shares the base N of the California models.
df = pd.read_csv(MERGED_ROOT / 'outputs' / 'study2' / 'california_regression_data.csv')

print(f'Sample: {len(df):,} cases')

# =============================================================================
# CROSS-TABULATION: COUNTS
# =============================================================================
print('\n--- Raw Counts ---')
ct = pd.crosstab(df['race_std'], df['wound_location_std'],
                 margins=True, margins_name='Total')
# Reorder columns
col_order = ['Head/Neck', 'Chest', 'Abdomen', 'Back', 'Extremities', 'Total']
ct = ct[[c for c in col_order if c in ct.columns]]
# Reorder rows
row_order = ['White', 'Black', 'Hispanic', 'Total']
ct = ct.reindex([r for r in row_order if r in ct.index])
print(ct)

# =============================================================================
# ROW PERCENTAGES (% of each race in each wound location)
# =============================================================================
print('\n--- Row Percentages (% within race) ---')
ct_pct = pd.crosstab(df['race_std'], df['wound_location_std'], normalize='index') * 100
ct_pct = ct_pct[[c for c in col_order if c in ct_pct.columns and c != 'Total']]
ct_pct = ct_pct.reindex([r for r in row_order if r in ct_pct.index and r != 'Total'])
print(ct_pct.round(1))

# =============================================================================
# CFR BY RACE x WOUND LOCATION
# =============================================================================
print('\n--- CFR by Race x Wound Location ---')
cfr_rows = []
for race in ['White', 'Black', 'Hispanic']:
    for wl in ['Head/Neck', 'Chest', 'Abdomen', 'Back', 'Extremities']:
        subset = df[(df['race_std'] == race) & (df['wound_location_std'] == wl)]
        n = len(subset)
        n_fatal = int(subset['fatal'].sum())
        cfr = n_fatal / n * 100 if n > 0 else np.nan
        cfr_rows.append({
            'race': race,
            'wound_location': wl,
            'n': n,
            'n_fatal': n_fatal,
            'cfr_pct': cfr,
        })

cfr_df = pd.DataFrame(cfr_rows)
cfr_pivot = cfr_df.pivot(index='race', columns='wound_location', values='cfr_pct')
cfr_pivot = cfr_pivot[[c for c in ['Head/Neck', 'Chest', 'Abdomen', 'Back', 'Extremities'] if c in cfr_pivot.columns]]
cfr_pivot = cfr_pivot.reindex(['White', 'Black', 'Hispanic'])
print(cfr_pivot.round(1))

# =============================================================================
# CHI-SQUARE TEST (reproduce for verification)
# =============================================================================
print('\n--- Chi-Square Test ---')
observed = pd.crosstab(df['race_std'], df['wound_location_std'])
chi2, p, dof, expected = stats.chi2_contingency(observed)
print(f'Chi-square({dof}) = {chi2:.2f}, p = {p:.3f}')
print(f'Interpretation: {"Significant" if p < 0.05 else "Not significant"} '
      f'difference in wound location by race')

# =============================================================================
# SAVE COMBINED TABLE
# =============================================================================
# Build a comprehensive table for the manuscript
combined_rows = []
for race in ['White', 'Black', 'Hispanic', 'Total']:
    for wl in ['Head/Neck', 'Chest', 'Abdomen', 'Back', 'Extremities']:
        if race == 'Total':
            subset = df[df['wound_location_std'] == wl]
        else:
            subset = df[(df['race_std'] == race) & (df['wound_location_std'] == wl)]
        n = len(subset)
        n_fatal = int(subset['fatal'].sum())
        cfr = n_fatal / n * 100 if n > 0 else np.nan

        # Row pct: what % of this race had this wound location
        if race == 'Total':
            race_total = len(df)
        else:
            race_total = len(df[df['race_std'] == race])
        row_pct = n / race_total * 100 if race_total > 0 else np.nan

        combined_rows.append({
            'race': race,
            'wound_location': wl,
            'n': n,
            'row_pct': round(row_pct, 1),
            'n_fatal': n_fatal,
            'cfr_pct': round(cfr, 1),
        })

combined_df = pd.DataFrame(combined_rows)
combined_df.to_csv(OUTPUT_DIR / 'race_wound_crosstab.csv', index=False)
print(f'\nSaved: race_wound_crosstab.csv')

# Also save the simple count cross-tab
ct.to_csv(OUTPUT_DIR / 'race_wound_crosstab_counts.csv')
print(f'Saved: race_wound_crosstab_counts.csv')

print('\n' + '=' * 70)
print('RACE x WOUND LOCATION CROSS-TABULATION COMPLETE')
print('=' * 70)

# --- Chi-square with and without the Back category (added 2026-10-01) ---
# The race difference lies in back wounds; report the test on the four front/head/limb
# categories so the manuscript can state that directly.
from scipy.stats import chi2_contingency  # noqa: E402
rows = []
for label, cats in [('five categories', ['Head/Neck', 'Chest', 'Abdomen', 'Back', 'Extremities']),
                    ('excluding back', ['Head/Neck', 'Chest', 'Abdomen', 'Extremities'])]:
    sub = df[df['wound_location_std'].isin(cats)]
    chi2, p, dof, _ = chi2_contingency(pd.crosstab(sub['race_std'], sub['wound_location_std']))
    rows.append({'test': label, 'chi2': chi2, 'df': dof, 'p_value': p, 'n': len(sub)})
pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'race_wound_chisq_variants.csv', index=False)
print(pd.DataFrame(rows))

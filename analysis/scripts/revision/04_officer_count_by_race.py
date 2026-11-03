"""
REVISION: Officer count by race descriptive analysis.

Addresses Reviewer 1: "Does officer count differ by race of subject?
Are multi-officer incidents disproportionately common for Black subjects?"

Outputs:
  outputs/revision/officer_count_by_race_tx.csv
  outputs/revision/officer_count_by_race_ca.csv
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
print('REVISION: OFFICER COUNT BY RACE')
print('=' * 70)


def analyze_officer_count_by_race(df, officer_col, state_name):
    """Compute officer count descriptives by race."""
    races = ['White', 'Black', 'Hispanic']
    df_sub = df[df['race_std'].isin(races) & df[officer_col].notna()].copy()

    rows = []
    race_groups = []
    for race in races:
        subset = df_sub[df_sub['race_std'] == race]
        n = len(subset)
        mean_oc = subset[officer_col].mean()
        sd_oc = subset[officer_col].std()
        median_oc = subset[officer_col].median()
        pct_multi = (subset[officer_col] >= 2).mean() * 100
        pct_3plus = (subset[officer_col] >= 3).mean() * 100

        rows.append({
            'race': race,
            'n': n,
            'mean_officer_count': mean_oc,
            'sd_officer_count': sd_oc,
            'median_officer_count': median_oc,
            'pct_multi_officer': pct_multi,
            'pct_3plus_officers': pct_3plus,
        })
        race_groups.append(subset[officer_col].values)

        print(f'  {race}: n={n:,}, mean={mean_oc:.2f} (SD={sd_oc:.2f}), '
              f'median={median_oc:.0f}, multi={pct_multi:.1f}%, 3+={pct_3plus:.1f}%')

    # ANOVA (one-way)
    f_stat, anova_p = stats.f_oneway(*race_groups)
    print(f'\n  One-way ANOVA: F = {f_stat:.3f}, p = {anova_p:.4f}')

    # Kruskal-Wallis (nonparametric)
    kw_stat, kw_p = stats.kruskal(*race_groups)
    print(f'  Kruskal-Wallis: H = {kw_stat:.3f}, p = {kw_p:.4f}')

    # Pairwise comparisons (White vs Black, White vs Hispanic, Black vs Hispanic)
    pairs = [('White', 'Black'), ('White', 'Hispanic'), ('Black', 'Hispanic')]
    for r1, r2 in pairs:
        g1 = df_sub[df_sub['race_std'] == r1][officer_col]
        g2 = df_sub[df_sub['race_std'] == r2][officer_col]
        t, p = stats.ttest_ind(g1, g2, equal_var=False)
        print(f'  {r1} vs {r2}: t = {t:.3f}, p = {p:.4f} (Welch)')

    # Chi-square test on multi-officer (2+) by race
    cross = pd.crosstab(df_sub['race_std'], df_sub[officer_col] >= 2)
    chi2, chi_p, dof, _ = stats.chi2_contingency(cross)
    print(f'  Chi-square (multi-officer x race): chi2 = {chi2:.3f}, p = {chi_p:.4f}')

    results_df = pd.DataFrame(rows)
    results_df['anova_F'] = f_stat
    results_df['anova_p'] = anova_p
    results_df['kruskal_H'] = kw_stat
    results_df['kruskal_p'] = kw_p
    results_df['chi2_multi_officer'] = chi2
    results_df['chi2_multi_officer_p'] = chi_p

    return results_df


# Texas
print('\n--- Texas ---')
tx = pd.read_csv(MERGED_ROOT / 'outputs' / 'study1' / 'texas_analysis_ready.csv')
tx_results = analyze_officer_count_by_race(tx, 'Officer_Count', 'Texas')
tx_results.to_csv(OUTPUT_DIR / 'officer_count_by_race_tx.csv', index=False)
print(f'\nSaved: officer_count_by_race_tx.csv')

# California
print('\n--- California ---')
ca = pd.read_csv(MERGED_ROOT / 'outputs' / 'study2' / 'california_analysis_ready.csv')
ca_results = analyze_officer_count_by_race(ca, 'num_involved_officers', 'California')
ca_results.to_csv(OUTPUT_DIR / 'officer_count_by_race_ca.csv', index=False)
print(f'\nSaved: officer_count_by_race_ca.csv')

print('\n' + '=' * 70)
print('OFFICER COUNT BY RACE ANALYSIS COMPLETE')
print('=' * 70)

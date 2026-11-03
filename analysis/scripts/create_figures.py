"""
Merged paper: Create publication-quality figures.

Figures:
  1. CFR by officer count (both states, side-by-side)
  2. Forest plot: Officer count coefficient across models (TX + CA + pooled)
  3. Robustness specification curve (Officer count coefficient across all specs)
  6. Killing cascade: (A) CFR heatmap with density contours, (B) CDF shift curves

Geographic maps (TX and CA county choropleths) are produced by
the geographic analysis scripts in each study folder.

Adapted from officer_count_paper figure scripts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
MERGED_ROOT = SCRIPT_DIR.parent
TX_TABLES = MERGED_ROOT / 'outputs' / 'study1' / 'tables'
CA_TABLES = MERGED_ROOT / 'outputs' / 'study2' / 'tables'
FIGURES_DIR = MERGED_ROOT / 'outputs' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Style settings. Figures are drawn at their printed size (6.5in text width, or the width
# set in the qmd), so font sizes here are the sizes on the page (2026-10-12: earlier
# versions were drawn at 10-18in and shrunk to fit, which printed text at 3-6pt).
TEXT_WIDTH = 6.5
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 9.5,
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'grid.alpha': 0.4,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


def save(fig, stem):
    """Save at exactly figsize (no tight bbox), so the qmd can place it at 100% scale."""
    fig.savefig(FIGURES_DIR / f'{stem}.png')
    fig.savefig(FIGURES_DIR / f'{stem}.pdf')
    plt.close(fig)


print('=' * 70)
print('MERGED PAPER: CREATING FIGURES')
print('=' * 70)

# =============================================================================
# LOAD DATA
# =============================================================================
print('\nLoading data...')

tx_cfr = pd.read_csv(TX_TABLES / 'cfr_by_officer_count.csv')
ca_cfr = pd.read_csv(CA_TABLES / 'cfr_by_officer_count.csv')

try:
    pooled_results = pd.read_csv(MERGED_ROOT / 'outputs' / 'pooled_regression_results.csv')
except FileNotFoundError:
    pooled_results = None
    print('  WARNING: pooled_regression_results.csv not found. Run pooled_analysis.py first.')

try:
    tx_robustness = pd.read_csv(TX_TABLES / 'robustness_results.csv')
except FileNotFoundError:
    tx_robustness = None

try:
    ca_robustness = pd.read_csv(CA_TABLES / 'robustness_results.csv')
except FileNotFoundError:
    ca_robustness = None

# =============================================================================
# FIGURE 1: CFR BY OFFICER COUNT (BOTH STATES)
# =============================================================================
print('\nCreating Figure 1: CFR by Officer Count (Both States)...')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.0), sharey=True,
                               layout='constrained')


def cfr_panel(ax, cfr, color, edge, title, xlabel):
    """Observed CFR bars with 95% Wald CIs; n sits under each tick so it cannot collide
    with the error bars. Counts with fewer than 10 cases are omitted."""
    d = cfr[cfr['total'] >= 10].copy()
    d['se'] = np.sqrt(d['cfr'] * (100 - d['cfr']) / d['total'])
    ax.bar(d['officers'], d['cfr'], color=color, alpha=0.7, edgecolor=edge, linewidth=0.6)
    ax.errorbar(d['officers'], d['cfr'], yerr=1.96 * d['se'], fmt='none', color='black',
                capsize=2, linewidth=0.8)
    ax.set_xticks(d['officers'])
    ax.set_xticklabels([f'{int(o)}\n({int(n):,})' for o, n in zip(d['officers'], d['total'])],
                       fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc='left')
    ax.set_ylim(0, 100)
    ax.grid(axis='x', visible=False)


# No trend line: an unweighted fit to cell means is not the regression estimate (PQ R1).
cfr_panel(ax1, tx_cfr, 'steelblue', 'navy', 'A. Texas', 'Officers who fired (n)')
cfr_panel(ax2, ca_cfr, 'forestgreen', 'darkgreen', 'B. California', 'Officers involved (n)')
ax1.set_ylabel('Case fatality (%)')
save(fig, 'fig1_cfr_by_officer_count')
print('  Saved Figure 1')

# =============================================================================
# FIGURE 2: FOREST PLOT (TX, CA, POOLED)
# =============================================================================
if pooled_results is not None:
    print('\nCreating Figure 2: Pooled Regression Forest Plot...')

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.3), layout='constrained')

    row_labels = {'Texas': 'Texas (M3)\nofficers who fired',
                  'California': 'California (M3)\nofficers involved',
                  'Pooled': 'Pooled (PM3)\nstate-by-year FE'}
    studies = [row_labels.get(s_, s_) for s_ in pooled_results['study']]
    coefs = pooled_results['Coef'].tolist()
    ci_lows = pooled_results['CI_lower'].tolist()
    ci_highs = pooled_results['CI_upper'].tolist()

    y_pos = np.arange(len(studies))
    colors = ['navy', 'navy', 'darkred']
    markers = ['o', 'o', 'D']
    sizes = [6, 6, 7]

    for i, (study, coef_val, ci_l, ci_h) in enumerate(zip(studies, coefs, ci_lows, ci_highs)):
        ax.errorbar(coef_val, i, xerr=[[coef_val - ci_l], [ci_h - coef_val]],
                    fmt=markers[i], color=colors[i], markersize=sizes[i],
                    capsize=3, capthick=1.2, linewidth=1.2)
        ax.annotate(f'{coef_val:.2f} pp [{ci_l:.2f}, {ci_h:.2f}]',
                    (ci_h + 0.15, i), fontsize=8.5, va='center')

    # Reference line at zero (no association in an LPM) and at the pooled estimate
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    ax.axvline(x=coefs[-1], color='darkred', linestyle=':', linewidth=0.8, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(studies, fontsize=9)
    ax.set_xlabel('Coefficient, pp per additional officer')
    ax.invert_yaxis()
    ax.set_ylim(len(studies) - 0.5, -0.5)
    ax.set_xlim(min(0, min(ci_lows)) - 0.5, max(ci_highs) + 2.6)
    save(fig, 'fig2_forest_plot')
    print('  Saved Figure 2')

# =============================================================================
# FIGURE 3: ROBUSTNESS SPECIFICATION CURVE
# =============================================================================
if tx_robustness is not None or ca_robustness is not None:
    print('\nCreating Figure 3: Robustness Specification Curve...')

    # Display-only label shortening so the label column fits a 6.5in page; specification
    # numbers (cited in the manuscript) are unchanged.
    SHORT_LABELS = {
        '1. Baseline (M4: full covariates)': '1. Baseline (M4)',
        '14. M3 covariates (no incident circumstances)': '14. M3 covariates (no circumstances)',
        '15. HC1 SEs, no year FE (Round 1 estimator)': '15. HC1 SEs, no year FE (Round 1)',
        '18. Without media-reported weapon controls': '18. No media-reported weapon controls',
        '19. TJI records only (no OAG-only cases)': '19. TJI records only',
        '21. Excluding OAG cases missing from TJI': '21. Excl. OAG cases missing from TJI',
        '1. Baseline (M6: full covariates + agency FE + wound location)': '1. Baseline (M6)',
        '8. Without wound location controls (M5)': '8. No wound location controls (M5)',
        '20. High-lethality wounds (Head/Chest)': '20. Head/neck or chest wounds only',
        '21. Low-lethality wounds (Abdomen/Back/Extremities)': '21. Other wound locations only',
        '22. HC1 SEs, no fixed effects (Round 1 estimator)': '22. HC1 SEs, no FE (Round 1)',
        '26. Without agency fixed effects (year FE only)': '26. No agency FE (year FE only)',
    }

    def spec_panel(ax, rob, coef, lo, hi, p, title):
        """Point and 95% CI per specification, sorted by coefficient (largest at top)."""
        d = rob.dropna(subset=[coef]).sort_values(coef).reset_index(drop=True)
        label_col = 'Specification' if 'Specification' in d.columns else d.columns[0]
        sig = d[p].fillna(1.0) < 0.05
        for mask, color in [(sig, 'navy'), (~sig, 'gray')]:
            sub = d[mask]
            ax.errorbar(sub[coef], sub.index,
                        xerr=[(sub[coef] - sub[lo]).fillna(0), (sub[hi] - sub[coef]).fillna(0)],
                        fmt='o', color=color, ecolor=color, markersize=3.5, capsize=1.5,
                        elinewidth=0.9, zorder=3)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=0.9, alpha=0.7)
        ax.set_yticks(d.index)
        ax.set_yticklabels([SHORT_LABELS.get(x, x) for x in d[label_col]], fontsize=8)
        ax.set_ylim(-0.7, len(d) - 0.3)
        ax.set_title(title, loc='left')
        return d

    n_tx = tx_robustness['Coef'].notna().sum()
    n_ca = ca_robustness['Officer_Coef'].notna().sum()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(TEXT_WIDTH, 6.7), sharex=True,  # 6.7in leaves room for the caption
                                   layout='constrained',
                                   gridspec_kw={'height_ratios': [n_tx + 1.5, n_ca + 1.5]})
    spec_panel(ax1, tx_robustness, 'Coef', 'CI_lower', 'CI_upper', 'p_value', 'A. Texas')
    spec_panel(ax2, ca_robustness, 'Officer_Coef', 'Officer_CI_lower', 'Officer_CI_upper',
               'Officer_p', 'B. California')
    ax2.set_xlabel('Officer-count coefficient (pp)')
    ax2.tick_params(axis='x', labelbottom=True)

    sig_handle = Line2D([0], [0], color='navy', marker='o', markersize=4, linewidth=1,
                        label='p < .05 (point and 95% CI)')
    nonsig_handle = Line2D([0], [0], color='gray', marker='o', markersize=4, linewidth=1,
                           label='p ≥ .05 (point and 95% CI)')
    fig.legend(handles=[sig_handle, nonsig_handle], loc='outside lower center', ncol=2)
    save(fig, 'fig3_robustness_curve')
    print('  Saved Figure 3')

# =============================================================================
# FIGURE 3 (file fig6_lethal_cascade): CA case fatality by officer count x regions wounded
# =============================================================================
# Revised 2026-10-01 (PQ R&R): single panel on the CA regression sample. The Round 1 panel B
# (simulated "lives saved" CDFs) is removed; model-based expected-death differences are in the
# appendix (revision/10_predicted_fatality.py). Cells are drawn without interpolation, and the
# y-axis is the number of distinct URSUS regions wounded, not a count of wounds.
print('\nCreating Figure 3: Case fatality by officers involved x regions wounded (CA)...')

ca_heat = pd.read_csv(MERGED_ROOT / 'outputs' / 'study2' / 'california_regression_data.csv')
ca_heat['off_bin'] = ca_heat['num_involved_officers'].clip(upper=5).astype(int)
ca_heat['wnd_bin'] = ca_heat['wound_count'].clip(upper=5).astype(int)
bins = [1, 2, 3, 4, 5]
cells = (ca_heat.groupby(['wnd_bin', 'off_bin'])['fatal'].agg(['mean', 'size'])
         .reindex(pd.MultiIndex.from_product([bins, bins], names=['wnd_bin', 'off_bin']))
         .fillna({'size': 0}))
cells.reset_index().rename(columns={'mean': 'cfr', 'size': 'n'}).to_csv(
    MERGED_ROOT / 'outputs' / 'figures' / 'fig3_heatmap_cells.csv', index=False)
cfr_grid = np.full((5, 5), np.nan)
count_grid = np.zeros((5, 5))
for (wb, ob), row in cells.iterrows():
    count_grid[wb - 1, ob - 1] = row['size']
    if row['size'] >= 5:
        cfr_grid[wb - 1, ob - 1] = row['mean'] * 100

HEATMAP_WIDTH = 5.5  # matches width="5.5in" in the qmd
fig, ax = plt.subplots(figsize=(HEATMAP_WIDTH, 4.0), layout='constrained')
im = ax.imshow(cfr_grid, cmap='RdYlBu_r', aspect='auto', vmin=30, vmax=90,
               interpolation='nearest', origin='lower', extent=[0.5, 5.5, 0.5, 5.5])
for i in range(5):
    for j in range(5):
        if not np.isnan(cfr_grid[i, j]):
            ax.text(j + 1, i + 1, f'{cfr_grid[i, j]:.0f}%\nn = {int(count_grid[i, j])}',
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    # white on the darkest reds, where black text loses contrast
                    color='white' if cfr_grid[i, j] >= 84 else 'black')
ax.grid(False)  # global style gridlines would cross the cell labels
ax.set_xticks(bins)
ax.set_xticklabels(['1', '2', '3', '4', '5+'], fontsize=9)
ax.set_yticks(bins)
ax.set_yticklabels(['1', '2', '3', '4', '5+'], fontsize=9)
ax.set_xlabel('Officers involved')
ax.set_ylabel('Anatomical regions wounded')
cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cbar.set_label('Observed case fatality (%)', fontsize=9)
cbar.ax.tick_params(labelsize=8.5)
save(fig, 'fig6_lethal_cascade')
print(f'  Saved Figure 3 (N = {int(count_grid.sum()):,})')

# =============================================================================
# SUMMARY
# =============================================================================
print('\n' + '=' * 70)
print('FIGURES COMPLETE')
print('=' * 70)
print(f'\nFigures saved to: {FIGURES_DIR}')

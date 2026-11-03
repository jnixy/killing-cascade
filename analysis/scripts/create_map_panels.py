"""
Merged paper: Create 3-panel county choropleth maps for Texas and California.

For each state, produces a figure with three panels:
  Panel A: Total gunshot incidents by county (YlOrRd)
  Panel B: Case Fatality Rate (%) for counties with n>=10 (RdPu)
  Panel C: % Black + Hispanic victims for counties with n>=10 (BuGn)

Counties with <10 incidents are grey in Panels B and C.

Data sources:
  - outputs/study1/tables/county_descriptives.csv (TX)
  - outputs/study2/tables/county_descriptives.csv (CA)
  - outputs/study1/texas_analysis_ready.csv (TX race data)
  - outputs/study2/california_analysis_ready.csv (CA race data)
  - Census TIGER/Line county shapefiles (downloaded automatically)
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from pathlib import Path
from adjustText import adjust_text
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
MERGED_ROOT = SCRIPT_DIR.parent
TX_TABLES = MERGED_ROOT / 'outputs' / 'study1' / 'tables'
CA_TABLES = MERGED_ROOT / 'outputs' / 'study2' / 'tables'
TX_DATA = MERGED_ROOT / 'outputs' / 'study1'
CA_DATA = MERGED_ROOT / 'outputs' / 'study2'
FIGURES_DIR = MERGED_ROOT / 'outputs' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Style settings (match create_figures.py). Drawn at printed size (6.5in wide), so font
# sizes are page sizes; the 2026-10-02 version was 17.6in wide and printed text at ~3pt.
TEXT_WIDTH = 6.5
# Stacked panels are height-limited (~2.3in per map), so a 6.5in canvas leaves the left half
# empty; the figure is drawn this wide and placed at width="3.9in" in the qmd.
MAP_FIG_WIDTH = 3.9
MAP_FIG_HEIGHT = 7.1  # figure + double-spaced caption must fit a 9in text block
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

GREY = '#d9d9d9'
GREY_EDGE = '#cccccc'
INCIDENT_THRESHOLD = 10

print('=' * 70)
print('MERGED PAPER: CREATING 3-PANEL COUNTY MAPS')
print('=' * 70)

# =============================================================================
# LOAD DATA
# =============================================================================
print('\nLoading county descriptives...')

tx_counties = pd.read_csv(TX_TABLES / 'county_descriptives.csv')
ca_counties = pd.read_csv(CA_TABLES / 'county_descriptives.csv')
print(f'  TX: {len(tx_counties)} counties')
print(f'  CA: {len(ca_counties)} counties')

# =============================================================================
# COMPUTE RACE COMPOSITION BY COUNTY
# =============================================================================
print('\nComputing % Black + Hispanic by county...')

# Texas
tx_df = pd.read_csv(TX_DATA / 'texas_analysis_ready.csv')
tx_df = tx_df[tx_df['incident_county'].notna()].copy()
tx_df['county_clean'] = tx_df['incident_county'].str.strip().str.title()
tx_df['is_bh'] = tx_df['race_std'].isin(['Black', 'Hispanic']).astype(int)
tx_race = tx_df.groupby('county_clean').agg(
    n_total=('is_bh', 'size'),
    n_bh=('is_bh', 'sum')
).reset_index()
tx_race['pct_black_hispanic'] = (tx_race['n_bh'] / tx_race['n_total']) * 100
tx_counties = tx_counties.merge(
    tx_race[['county_clean', 'pct_black_hispanic']],
    left_on='county', right_on='county_clean', how='left'
).drop(columns='county_clean')

# California
ca_df = pd.read_csv(CA_DATA / 'california_analysis_ready.csv')
ca_df = ca_df[ca_df['county'].notna()].copy()
ca_df['county_clean'] = (
    ca_df['county'].str.strip().str.title()
    .str.replace(r'\s+County$', '', regex=True)
)
ca_df['is_bh'] = ca_df['race_std'].isin(['Black', 'Hispanic']).astype(int)
ca_race = ca_df.groupby('county_clean').agg(
    n_total=('is_bh', 'size'),
    n_bh=('is_bh', 'sum')
).reset_index()
ca_race['pct_black_hispanic'] = (ca_race['n_bh'] / ca_race['n_total']) * 100
ca_counties = ca_counties.merge(
    ca_race[['county_clean', 'pct_black_hispanic']],
    on='county_clean', how='left'
)

print(f'  TX: race data for {tx_counties["pct_black_hispanic"].notna().sum()} counties')
print(f'  CA: race data for {ca_counties["pct_black_hispanic"].notna().sum()} counties')

# =============================================================================
# LOAD SHAPEFILES
# =============================================================================
print('\nLoading county shapefiles from Census TIGER/Line...')

us_counties_url = 'https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip'
us_counties = gpd.read_file(us_counties_url)

tx_shapes = us_counties[us_counties['STATEFP'] == '48'].copy()
ca_shapes = us_counties[us_counties['STATEFP'] == '06'].copy()
print(f'  TX shapefile: {len(tx_shapes)} counties')
print(f'  CA shapefile: {len(ca_shapes)} counties')

# =============================================================================
# MERGE DATA WITH SHAPEFILES
# =============================================================================
print('\nMerging data with shapefiles...')

# TX
tx_shapes['county_name'] = tx_shapes['NAME'].str.strip().str.title()
tx_merged = tx_shapes.merge(tx_counties, left_on='county_name', right_on='county', how='left')
tx_merged['n_incidents'] = tx_merged['n_incidents'].fillna(0).astype(int)

# CA
ca_shapes['county_name'] = ca_shapes['NAME'].str.strip().str.title()
ca_merged = ca_shapes.merge(ca_counties, left_on='county_name', right_on='county_clean', how='left')
ca_merged['n_incidents'] = ca_merged['n_incidents'].fillna(0).astype(int)

tx_matched = (tx_merged['n_incidents'] > 0).sum()
ca_matched = (ca_merged['n_incidents'] > 0).sum()
print(f'  TX: {tx_matched} counties with incidents')
print(f'  CA: {ca_matched} counties with incidents')


# =============================================================================
# PLOTTING FUNCTION
# =============================================================================

def create_map_panels(gdf, state_name, year_range, top5_counties, county_key,
                      total_n, total_counties, filename, label_positions=None):
    """Create a 3-panel choropleth figure for one state."""

    has_data = gdf[gdf['n_incidents'] > 0]
    no_data = gdf[gdf['n_incidents'] == 0]
    above_threshold = gdf[gdf['n_incidents'] >= INCIDENT_THRESHOLD]
    below_threshold = gdf[gdf['n_incidents'] < INCIDENT_THRESHOLD]

    # Panels stacked in one column so each map is ~2.3in tall, wider than in a row
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(MAP_FIG_WIDTH, MAP_FIG_HEIGHT), layout='constrained')

    def add_colorbar(sm, ax, label):
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=18, pad=0.02)
        cbar.set_label(label, fontsize=9)
        cbar.ax.tick_params(labelsize=8.5)
        return cbar

    # --- Panel A: Total Incidents ---
    max_incidents = has_data['n_incidents'].max()
    norm_a = Normalize(vmin=0, vmax=max_incidents)
    cmap_a = plt.cm.YlOrRd

    no_data.plot(ax=ax1, color='#f5f5dc', edgecolor=GREY_EDGE, linewidth=0.3)
    has_data.plot(column='n_incidents', ax=ax1, cmap=cmap_a, norm=norm_a,
                  edgecolor='#999999', linewidth=0.3)

    sm_a = plt.cm.ScalarMappable(cmap=cmap_a, norm=norm_a)
    sm_a.set_array([])
    add_colorbar(sm_a, ax1, 'Civilians shot')

    # Label top 5 counties. label_positions (lon, lat) places callouts by hand where
    # adjust_text would leave them on top of the county (CA's small coastal counties).
    texts = []
    for county_name in top5_counties:
        row = gdf[gdf['county_name'] == county_name]
        if len(row) == 0:
            continue
        count = int(row['n_incidents'].iloc[0])
        centroid = row.geometry.centroid.iloc[0]
        xytext = (label_positions or {}).get(county_name, (centroid.x, centroid.y))
        texts.append(
            ax1.annotate(f'{county_name}\n({count})',
                         xy=(centroid.x, centroid.y), xytext=xytext,
                         fontsize=8, fontweight='bold',
                         ha='center', va='center',
                         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray',
                                   alpha=0.85, lw=0.5),
                         arrowprops=dict(arrowstyle='-', color='gray', lw=0.6))
        )
    if texts and not label_positions:
        adjust_text(texts, ax=ax1)

    ax1.set_title('A. Civilians shot', loc='left')
    ax1.axis('off')

    # --- Panel B: Case Fatality Rate ---
    cfr_vals = above_threshold['cfr'].dropna()
    if len(cfr_vals) > 0:
        vmin_b = max(cfr_vals.min() - 5, 0)
        vmax_b = min(cfr_vals.max() + 5, 100)
    else:
        vmin_b, vmax_b = 30, 90
    norm_b = Normalize(vmin=vmin_b, vmax=vmax_b)
    cmap_b = plt.cm.RdPu

    below_threshold.plot(ax=ax2, color=GREY, edgecolor=GREY_EDGE, linewidth=0.3)
    above_threshold.plot(column='cfr', ax=ax2, cmap=cmap_b, norm=norm_b,
                         edgecolor='#999999', linewidth=0.3)

    sm_b = plt.cm.ScalarMappable(cmap=cmap_b, norm=norm_b)
    sm_b.set_array([])
    add_colorbar(sm_b, ax2, 'Case fatality (%)')

    # the >=10-case restriction for panels B and C is stated in the caption
    ax2.set_title('B. Case fatality', loc='left')
    ax2.axis('off')

    # --- Panel C: % Black + Hispanic ---
    bh_vals = above_threshold['pct_black_hispanic'].dropna()
    if len(bh_vals) > 0:
        vmin_c = max(bh_vals.min() - 5, 0)
        vmax_c = min(bh_vals.max() + 5, 100)
    else:
        vmin_c, vmax_c = 20, 90
    norm_c = Normalize(vmin=vmin_c, vmax=vmax_c)
    cmap_c = plt.cm.BuGn

    below_threshold.plot(ax=ax3, color=GREY, edgecolor=GREY_EDGE, linewidth=0.3)
    above_threshold.plot(column='pct_black_hispanic', ax=ax3, cmap=cmap_c, norm=norm_c,
                         edgecolor='#999999', linewidth=0.3)

    sm_c = plt.cm.ScalarMappable(cmap=cmap_c, norm=norm_c)
    sm_c.set_array([])
    add_colorbar(sm_c, ax3, 'Black or Hispanic (%)')

    ax3.set_title('C. Black or Hispanic civilians', loc='left')
    ax3.axis('off')

    # No suptitle or N footer: the manuscript caption carries the title and the gray-county rule.
    # Saved at exactly figsize so the qmd places it at 100% scale.
    fig.savefig(FIGURES_DIR / f'{filename}.png')
    fig.savefig(FIGURES_DIR / f'{filename}.pdf')
    plt.close()
    print(f'  Saved: {filename}.png and .pdf')


# =============================================================================
# CREATE FIGURES
# =============================================================================

# Texas
tx_total_n = int(tx_merged[tx_merged['n_incidents'] > 0]['n_incidents'].sum())
tx_total_counties = (tx_merged['n_incidents'] > 0).sum()
print(f'\nCreating Texas 3-panel map (N={tx_total_n:,}, {tx_total_counties} counties)...')

create_map_panels(
    gdf=tx_merged,
    state_name='Texas',
    year_range='2015\u20132025',
    top5_counties=['Harris', 'Bexar', 'Dallas', 'Tarrant', 'Travis'],
    county_key='county',
    total_n=tx_total_n,
    total_counties=tx_total_counties,
    filename='fig_texas_map_panels',
    # offset from each county (Harris into the Gulf) so no callout covers its own county
    label_positions={'Harris': (-94.2, 28.3), 'Bexar': (-100.6, 28.2),
                     'Travis': (-99.6, 31.4), 'Dallas': (-94.9, 33.9),
                     'Tarrant': (-98.9, 34.1)}
)

# California
ca_total_n = int(ca_merged[ca_merged['n_incidents'] > 0]['n_incidents'].sum())
ca_total_counties = (ca_merged['n_incidents'] > 0).sum()
print(f'\nCreating California 3-panel map (N={ca_total_n:,}, {ca_total_counties} counties)...')

create_map_panels(
    gdf=ca_merged,
    state_name='California',
    year_range='2016\u20132025',
    top5_counties=['Los Angeles', 'San Bernardino', 'Riverside', 'Orange', 'San Diego'],
    county_key='county_clean',
    total_n=ca_total_n,
    total_counties=ca_total_counties,
    filename='fig_california_map_panels',
    # coastal counties labeled out over the ocean, inland ones in the empty Nevada corner
    label_positions={'Los Angeles': (-121.9, 35.7), 'Orange': (-121.1, 33.9),
                     'San Diego': (-119.5, 32.1), 'San Bernardino': (-115.9, 38.1),
                     'Riverside': (-114.5, 36.2)}
)

# =============================================================================
# SUMMARY
# =============================================================================
print('\n' + '=' * 70)
print('3-PANEL COUNTY MAPS COMPLETE')
print('=' * 70)

tx_above = (tx_merged['n_incidents'] >= INCIDENT_THRESHOLD).sum()
ca_above = (ca_merged['n_incidents'] >= INCIDENT_THRESHOLD).sum()
print(f'\nTexas: {tx_total_n:,} incidents, {tx_total_counties} counties ({tx_above} with n>={INCIDENT_THRESHOLD})')
print(f'California: {ca_total_n:,} incidents, {ca_total_counties} counties ({ca_above} with n>={INCIDENT_THRESHOLD})')
print(f'\nFigures saved to: {FIGURES_DIR}')

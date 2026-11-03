"""
Study 2 - California: County-level geographic analysis of OIS incidents.

NEW analysis for the merged manuscript (Nix & Adams).
Produces county-level descriptive tables and choropleth maps.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
MERGED_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = MERGED_ROOT / 'outputs' / 'study2'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = MERGED_ROOT / 'outputs' / 'figures'
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 70)
print('MERGED PAPER - STUDY 2: GEOGRAPHIC ANALYSIS')
print('California County-Level Descriptives')
print('=' * 70)

# =============================================================================
# LOAD DATA
# =============================================================================
df = pd.read_csv(OUTPUT_DIR / 'california_analysis_ready.csv')
print(f'Loaded {len(df):,} OIS cases')

# Check county field
print(f'\nCounty field available: {"county" in df.columns}')
n_with_county = df['county'].notna().sum()
n_missing = df['county'].isna().sum()
print(f'Records with county: {n_with_county:,} ({n_with_county/len(df)*100:.1f}%)')
print(f'Records missing county: {n_missing:,}')

# =============================================================================
# COUNTY-LEVEL DESCRIPTIVES
# =============================================================================
print('\n' + '=' * 70)
print('COUNTY-LEVEL DESCRIPTIVE STATISTICS')
print('=' * 70)

# Filter to records with county data
df_county = df[df['county'].notna()].copy()

# Standardize county names (strip trailing "County" to consolidate city/county dupes
# e.g., "Los Angeles" and "Los Angeles County" -> "Los Angeles")
df_county['county_clean'] = (df_county['county'].str.strip().str.title()
                              .str.replace(r'\s+County$', '', regex=True))

# Aggregate by county
county_stats = df_county.groupby('county_clean').agg(
    n_incidents=('fatal', 'count'),
    n_fatal=('fatal', 'sum'),
    mean_officer_count=('num_involved_officers', 'mean'),
    median_officer_count=('num_involved_officers', 'median')
).reset_index()

county_stats['cfr'] = county_stats['n_fatal'] / county_stats['n_incidents'] * 100
county_stats['pct_of_total'] = county_stats['n_incidents'] / len(df_county) * 100

# Sort by incident count
county_stats = county_stats.sort_values('n_incidents', ascending=False).reset_index(drop=True)

# Top 20 counties
print('\nTop 20 Counties by OIS Volume:')
print(f'{"County":<25} {"N":<8} {"Fatal":<8} {"CFR%":<8} {"Mean Officers":<15} {"% of Total":<10}')
print('-' * 75)
top20 = county_stats.head(20)
for _, row in top20.iterrows():
    print(f'{row["county_clean"]:<25} {row["n_incidents"]:<8.0f} {row["n_fatal"]:<8.0f} '
          f'{row["cfr"]:<8.1f} {row["mean_officer_count"]:<15.2f} {row["pct_of_total"]:<10.1f}')

# Concentration statistics
n_counties_total = len(county_stats)
top5_n = county_stats.head(5)['n_incidents'].sum()
top10_n = county_stats.head(10)['n_incidents'].sum()
total_n = county_stats['n_incidents'].sum()

print(f'\n--- Geographic Concentration ---')
print(f'Total counties with OIS: {n_counties_total}')
print(f'Top 5 counties: {top5_n:,} incidents ({top5_n/total_n*100:.1f}% of total)')
print(f'Top 10 counties: {top10_n:,} incidents ({top10_n/total_n*100:.1f}% of total)')

# Save county-level table
county_stats.to_csv(TABLES_DIR / 'county_descriptives.csv', index=False)
print(f'\nSaved: county_descriptives.csv ({len(county_stats)} counties)')

# Save concentration summary for inline code
concentration = {
    'n_counties': n_counties_total,
    'top5_incidents': top5_n,
    'top5_pct': top5_n / total_n * 100,
    'top10_incidents': top10_n,
    'top10_pct': top10_n / total_n * 100,
    'total_incidents_with_county': total_n
}
pd.DataFrame([concentration]).to_csv(TABLES_DIR / 'county_concentration.csv', index=False)

# =============================================================================
# CHOROPLETH MAP
# =============================================================================
print('\n' + '=' * 70)
print('GENERATING CHOROPLETH MAP')
print('=' * 70)

try:
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Download California county shapefile from Census Bureau
    ca_counties_url = 'https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip'

    print('Loading US county shapefile...')
    us_counties = gpd.read_file(ca_counties_url)

    # Filter to California (STATEFP == '06')
    ca_counties = us_counties[us_counties['STATEFP'] == '06'].copy()
    print(f'California counties in shapefile: {len(ca_counties)}')

    # Merge with our data
    ca_counties['county_name'] = ca_counties['NAME'].str.strip().str.title()
    county_map = ca_counties.merge(county_stats, left_on='county_name',
                                   right_on='county_clean', how='left')
    county_map['n_incidents'] = county_map['n_incidents'].fillna(0)

    # Create choropleth
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    # Use log scale for better visualization
    county_map['n_incidents_log'] = np.log1p(county_map['n_incidents'])

    county_map.plot(
        column='n_incidents_log',
        ax=ax,
        legend=True,
        legend_kwds={'label': 'OIS Incidents (log scale)', 'shrink': 0.6},
        cmap='YlOrRd',
        edgecolor='gray',
        linewidth=0.3,
        missing_kwds={'color': 'lightgray', 'label': 'No incidents'}
    )

    ax.set_title('Officer-Involved Shootings by County\nCalifornia, 2016-2025', fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'california_county_map.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'california_county_map.pdf', bbox_inches='tight')
    plt.close()
    print(f'Saved: california_county_map.png and .pdf')

except ImportError:
    print('\nWARNING: geopandas not installed. Skipping choropleth map.')
    print('Install with: pip install geopandas')
    print('County descriptive tables were still saved successfully.')

except Exception as e:
    print(f'\nWARNING: Map generation failed: {e}')
    print('County descriptive tables were still saved successfully.')

print('\n' + '=' * 70)
print('GEOGRAPHIC ANALYSIS COMPLETE')
print('=' * 70)

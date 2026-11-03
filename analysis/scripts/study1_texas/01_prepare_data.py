"""
Study 1 - Texas: Prepare officer count analysis dataset, 2015-2025.

Data sources:
  - Texas Justice Initiative (TJI) compilation of Texas OAG officer-involved shooting reports,
    incidents through 2024-08-26 (https://texasjusticeinitiative.org/)
  - Texas OAG reports downloaded directly for incidents 2024-08-27 through 2025-12-31, and
    used to verify and correct TJI officer counts and duplicate rows throughout
Input file: data/processed/texas_ois_2015_2025.csv, built by 00, 00b, and 00c (run those first).

This script is adapted from officer_count_paper/Study 1 - Texas/scripts/01_prepare_data.py
for the merged manuscript (Nix & Adams).
"""

import os
import re

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
MERGED_ROOT = SCRIPT_DIR.parent.parent
PROJECT_ROOT = MERGED_ROOT.parent

DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
# TX_OIS_FILE / TX_OUTPUT_DIR let revision/18_tx_linkage_sensitivity.py rerun this script on an
# alternative input without touching the primary outputs; both default to the primary paths.
OUTPUT_DIR = Path(os.environ.get('TX_OUTPUT_DIR', MERGED_ROOT / 'outputs' / 'study1'))
TABLES_DIR = OUTPUT_DIR / 'tables'

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 70)
print('MERGED PAPER - STUDY 1: DATA PREPARATION')
print('Texas OIS Data (TJI + OAG reports), 2015-2025')
print('=' * 70)

# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================
print('\n' + '=' * 70)
print('SECTION 1: LOADING DATA')
print('=' * 70)

df = pd.read_csv(os.environ.get('TX_OIS_FILE', DATA_DIR / 'texas_ois_2015_2025.csv'), low_memory=False)
print(f'Loaded Texas OIS data: {len(df):,} total records')
print(df['source'].value_counts().to_string())
print(df['officer_count_source'].value_counts().to_string())

df['date_incident'] = pd.to_datetime(df['date_incident'], errors='coerce')
print(f'Date range: {df["date_incident"].min().date()} to {df["date_incident"].max().date()}')

# =============================================================================
# SECTION 2: CREATE STANDARD VARIABLE NAMES
# =============================================================================
print('\n' + '=' * 70)
print('SECTION 2: MAPPING VARIABLES')
print('=' * 70)

ois_df = pd.DataFrame()

# --- Outcome variable: Fatal ---
print('\nOutcome variable (civilian_died):')
print(df['civilian_died'].value_counts())
ois_df['fatal'] = (df['civilian_died'] == 'DEATH').astype(int)
print(f'\nFatal incidents: {ois_df["fatal"].sum():,} ({ois_df["fatal"].mean()*100:.1f}%)')

# --- Exposure: Officer count ---
print('\n--- Officer Count ---')
ois_df['Officer_Count'] = df['num_officers_recorded']
print(f'Officer count distribution:')
print(ois_df['Officer_Count'].value_counts().sort_index().head(10))

n_with_count = ois_df['Officer_Count'].notna().sum()
n_missing = ois_df['Officer_Count'].isna().sum()
print(f'\nRecords with officer count: {n_with_count:,} ({n_with_count/len(ois_df)*100:.1f}%)')

def categorize_officer_count(n):
    if pd.isna(n):
        return 'Unknown'
    n = int(n)
    if n == 1:
        return '1 officer'
    elif n == 2:
        return '2 officers'
    elif n == 3:
        return '3 officers'
    elif n <= 5:
        return '4-5 officers'
    else:
        return '6+ officers'

ois_df['officer_count_cat'] = ois_df['Officer_Count'].apply(categorize_officer_count)
print('\nOfficer count categories:')
print(ois_df['officer_count_cat'].value_counts())

# OAG form item: were multiple officers involved? TRUE with one recorded (firing) officer means
# other officers were present but did not fire -- evidence the count is shooters, not presence.
ois_df['multiple_officers_present'] = df['multiple_officers_involved'].astype(str).str.upper().eq('TRUE').astype(int)
ois_df['multiple_officers'] = (ois_df['Officer_Count'] >= 2).astype(float)
ois_df.loc[ois_df['Officer_Count'].isna(), 'multiple_officers'] = np.nan

# --- Victim Demographics ---
print('\n--- Victim Demographics ---')

ois_df['age_numeric'] = pd.to_numeric(df['civilian_age'], errors='coerce')
print(f'Victim Age: mean = {ois_df["age_numeric"].mean():.1f}, median = {ois_df["age_numeric"].median():.0f}')

ois_df['Victim_Gender'] = df['civilian_gender']
print('\nVictim Gender:')
print(ois_df['Victim_Gender'].value_counts())

print('\nVictim Race (raw):')
print(df['civilian_race'].value_counts())

race_map = {
    'WHITE': 'White',
    'BLACK': 'Black',
    'HISPANIC': 'Hispanic',
    'OTHER': 'Other'
}
ois_df['race_std'] = df['civilian_race'].map(race_map).fillna('Unknown')
ois_df['Victim_Race'] = ois_df['race_std']
print('\nVictim Race (standardized):')
print(ois_df['race_std'].value_counts())

# --- Weapon status ---
print('\n--- Weapon Status ---')
print('deadly_weapon values:')
print(df['deadly_weapon'].value_counts(dropna=False))

ois_df['had_weapon'] = df['deadly_weapon'].astype(int)
ois_df['Victim_Had_Weapon'] = df['deadly_weapon'].map({True: 'Yes - Deadly weapon', False: 'No - Unarmed'})
armed_pct = ois_df['had_weapon'].mean() * 100
print(f'\nVictim had weapon: {ois_df["had_weapon"].sum():,} ({armed_pct:.1f}%)')
print(f'Victim unarmed: {(ois_df["had_weapon"]==0).sum():,} ({100-armed_pct:.1f}%)')

# --- Officer demographics (first officer) ---
print('\n--- Officer Demographics (first officer) ---')

ois_df['Officer_Gender'] = df['officer_gender_1']
print('\nOfficer Gender:')
print(ois_df['Officer_Gender'].value_counts(dropna=False))

officer_race_map = {
    'WHITE': 'White',
    'BLACK': 'Black',
    'HISPANIC': 'Hispanic',
    'ASIAN': 'Other',
    'OTHER': 'Other'
}
ois_df['officer_race_std'] = df['officer_race_1'].map(officer_race_map).fillna('Unknown')
ois_df['Officer_Race'] = ois_df['officer_race_std']
print('\nOfficer Race (standardized):')
print(ois_df['officer_race_std'].value_counts())

ois_df['officer_off_duty'] = df['officer_on_duty'].map({True: 0, False: 1}).fillna(0).astype(int)
ois_df['Officer_Duty_Status'] = df['officer_on_duty'].map({True: 'On Duty', False: 'Off Duty'}).fillna('Unknown')

ois_df['officer_age_numeric'] = pd.to_numeric(df['officer_age_1'], errors='coerce')
ois_df['officer_female'] = (ois_df['Officer_Gender'] == 'FEMALE').astype(int)
ois_df['officer_black'] = (ois_df['officer_race_std'] == 'Black').astype(int)
ois_df['officer_hispanic'] = (ois_df['officer_race_std'] == 'Hispanic').astype(int)

ois_df['same_race_dyad'] = (ois_df['race_std'] == ois_df['officer_race_std']).astype(int)

# --- Date/time and geography ---
ois_df['Date_of_Incident'] = df['date_incident']
ois_df['incident_year'] = df['date_incident'].dt.year
print('\nIncidents by year:')
print(ois_df['incident_year'].value_counts().sort_index())

ois_df['incident_county'] = df['incident_county']
ois_df['incident_city'] = df['incident_city']
ois_df['agency_name'] = df['agency_name_1']
ois_df['ois_report_no'] = df['ois_report_no']
# Provenance: TJI vs OAG-only rows, and where each officer count came from (see 00c).
ois_df['source'] = df['source']
ois_df['officer_count_source'] = df['officer_count_source']
ois_df['tji_officer_count'] = df['tji_officer_count']

ois_df['Incident_Type'] = df['incident_result_of']

def categorize_incident(x):
    if pd.isna(x):
        return 'Unknown'
    x = str(x).lower()
    # The OAG form's answer labels changed in 2019. Map the 2015-18 labels to their 2019+
    # wording first; otherwise "other - specify type of call" matches 'call' and
    # "hostage, barricade, or other emergency situation" matches 'emergency', and both land
    # in Emergency Call for 2015-18 only.
    x = re.sub(r'other\s*-\s*specify type of call', 'other', x)
    x = re.sub(r'hostage,? barricade,? or other emergency situation|hostage/barricade/other emergency situation',
               'hostage/barricade/other emerg situation', x)
    if 'emergency' in x or 'call' in x or 'request' in x:
        return 'Emergency Call'
    elif 'traffic' in x or 'stop' in x:
        return 'Traffic Stop'
    elif 'warrant' in x:
        return 'Warrant Service'
    elif 'pursuit' in x or 'chase' in x:
        return 'Pursuit'
    elif 'domestic' in x or 'family' in x:
        return 'Domestic'
    elif 'assist' in x:
        return 'Assistance'
    else:
        return 'Other'

ois_df['incident_type_cat'] = ois_df['Incident_Type'].apply(categorize_incident)
print('\nIncident type categories:')
print(ois_df['incident_type_cat'].value_counts())

# --- Incident circumstances (added 2026-10-01, PQ R&R) ---
# Media-reported weapon category; "(DETAILS MISSING)" and blank are kept as their own category
# rather than dropped, since missingness is common (57%).
def weapon_media_cat(x):
    if pd.isna(x) or 'MISSING' in str(x).upper():
        return 'Missing'
    return {'FIREARM': 'Firearm', 'KNIFE/CUTTING': 'Knife/cutting',
            'VEHICLE': 'Vehicle'}.get(str(x).upper().strip(), 'Other')


ois_df['weapon_media_cat'] = df['weapon_reported_by_media_category'].apply(weapon_media_cat)
ois_df['multi_civilian'] = (df['num_rows_about_this_incident'].fillna(1) > 1).astype(int)

# Officer shot by a civilian in the same incident: Texas agencies file a separate OAG
# "Injury of Officer" report (CCP Art. 2.1395) when a civilian shoots an officer. Agency names
# are formatted differently in the two files, so reports are linked on incident date + county.
# The portal holds no reports for 2015 incidents, so the flag is structurally 0 for 2015 cases
# (absorbed by year fixed effects; noted in Methods). Reports re-parsed by 00c (2026-10-02).
inj = pd.read_csv(DATA_DIR / 'tx_oag_injury_of_officer.csv', low_memory=False)
inj_keys = set(zip(pd.to_datetime(inj['Date_of_Incident'], errors='coerce').dt.date,
                   inj['Incident_County'].astype(str).str.upper().str.strip()))
civ_keys = zip(df['date_incident'].dt.date, df['incident_county'].astype(str).str.upper().str.strip())
ois_df['officer_shot_by_civ'] = [int(k in inj_keys) for k in civ_keys]
print(f'\nOfficer shot by civilian (linked OAG report): {ois_df["officer_shot_by_civ"].sum():,}')
print('Weapon (media) categories:')
print(ois_df['weapon_media_cat'].value_counts())

# =============================================================================
# SECTION 3: PRELIMINARY OUTCOME ANALYSIS
# =============================================================================
print('\n' + '=' * 70)
print('SECTION 3: PRELIMINARY OUTCOME ANALYSIS')
print('=' * 70)

print('\nCase Fatality Rate by Officer Count (unadjusted):')
for cat in ['1 officer', '2 officers', '3 officers', '4-5 officers', '6+ officers']:
    subset = ois_df[ois_df['officer_count_cat'] == cat]
    if len(subset) > 0:
        n = len(subset)
        fatal = subset['fatal'].sum()
        rate = fatal / n * 100
        print(f'  {cat}: {fatal}/{n} = {rate:.1f}%')

print('\nSingle vs Multiple Officers:')
single = ois_df[ois_df['Officer_Count'] == 1]
multiple = ois_df[ois_df['Officer_Count'] >= 2]
if len(single) > 0 and len(multiple) > 0:
    print(f'  1 officer: {single["fatal"].sum()}/{len(single)} = {single["fatal"].mean()*100:.1f}%')
    print(f'  2+ officers: {multiple["fatal"].sum()}/{len(multiple)} = {multiple["fatal"].mean()*100:.1f}%')

# =============================================================================
# SECTION 4: SAVE ANALYSIS-READY DATASET
# =============================================================================
print('\n' + '=' * 70)
print('SECTION 4: SAVING ANALYSIS DATASET')
print('=' * 70)

output_path = OUTPUT_DIR / 'texas_analysis_ready.csv'
ois_df.to_csv(output_path, index=False)
print(f'\nSaved: {output_path}')
print(f'  Total OIS records: {len(ois_df):,}')
print(f'  Records with officer count: {n_with_count:,}')

# Save CFR summary table
cfr_data = []
for cat in ['1 officer', '2 officers', '3 officers', '4-5 officers', '6+ officers']:
    subset = ois_df[ois_df['officer_count_cat'] == cat]
    if len(subset) > 0:
        cfr_data.append({
            'officer_category': cat,
            'fatal': subset['fatal'].sum(),
            'total': len(subset),
            'cfr': subset['fatal'].mean() * 100
        })
cfr_df = pd.DataFrame(cfr_data)
cfr_df.to_csv(TABLES_DIR / 'cfr_by_officer_count.csv', index=False)
print(f'Saved: CFR by officer count table')

print('\n' + '=' * 70)
print('DATA PREPARATION COMPLETE')
print('=' * 70)
print('\nNext step: Run 02_run_analysis.py for statistical models')

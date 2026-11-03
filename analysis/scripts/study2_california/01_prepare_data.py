"""
Study 2 - California: Prepare unified analysis dataset.

Merges wound location analysis (from wound_race_paper) with officer count data
(from officer_count_paper) into a single analysis-ready dataset.

Data source: CA DOJ URSUS (Use of Force Reporting System), 2016-2025

Steps:
  1. Process raw URSUS data to identify gunshot cases
  2. Standardize variables (race, age, gender, wound location, resistance)
  3. Compute wound count metrics
  4. Harmonize incident IDs across reporting eras (2016-2021, 2022, 2023-2025)
  5. Merge incident-level officer counts
  6. Save unified analysis-ready dataset

Adapted from:
  - wound_race_paper/scripts/regenerate_analysis.py
  - officer_count_paper/Study 2 - California/scripts/01_prepare_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
MERGED_ROOT = SCRIPT_DIR.parent.parent
PROJECT_ROOT = MERGED_ROOT.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = MERGED_ROOT / 'outputs' / 'study2'
TABLES_DIR = OUTPUT_DIR / 'tables'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 70)
print('MERGED PAPER - STUDY 2: DATA PREPARATION')
print('California DOJ URSUS Data, 2016-2025')
print('=' * 70)

# =============================================================================
# SECTION 1: LOAD RAW DATA
# =============================================================================
print('\n' + '=' * 70)
print('SECTION 1: LOADING RAW DATA')
print('=' * 70)

civilians = pd.read_csv(DATA_DIR / 'master_civilian_data_clean.csv')
incidents = pd.read_csv(DATA_DIR / 'master_incident_data_clean.csv')
agencies = pd.read_csv(DATA_DIR / 'master_ori_agency_mapping.csv')

print(f'Total civilians: {len(civilians):,}')
print(f'Total incidents: {len(incidents):,}')

# =============================================================================
# SECTION 2: IDENTIFY GUNSHOT CASES
# =============================================================================
print('\n' + '=' * 70)
print('SECTION 2: IDENTIFYING GUNSHOT CASES')
print('=' * 70)

force_upper = civilians['received_force_type'].str.upper().fillna('')
hit_mask = force_upper.str.contains('FIREARM.*HIT|HIT.*FIREARM', na=False, regex=True)

gunshot_df = civilians[hit_mask].copy()
print(f'Gunshot cases identified: {len(gunshot_df):,}')
print('\nBy year:')
print(gunshot_df['data_year'].value_counts().sort_index())

# =============================================================================
# SECTION 3: STANDARDIZE VARIABLES
# =============================================================================
print('\n' + '=' * 70)
print('SECTION 3: STANDARDIZING VARIABLES')
print('=' * 70)

# --- Fatality outcome ---
gunshot_df['injury_level_upper'] = gunshot_df['injury_level'].str.upper().fillna('')
gunshot_df['fatal'] = gunshot_df['injury_level_upper'].str.contains('DEATH', na=False).astype(int)
print(f'Fatal: {gunshot_df["fatal"].sum():,} ({gunshot_df["fatal"].mean()*100:.1f}%)')


# --- Race standardization ---
def standardize_race(race):
    if pd.isna(race):
        return 'Unknown'
    race_upper = str(race).upper().strip()
    if ',' in race_upper:
        race_upper = race_upper.split(',')[0].strip()
    if 'HISPANIC' in race_upper or 'LATINO' in race_upper:
        return 'Hispanic'
    elif 'BLACK' in race_upper or 'AFRICAN' in race_upper:
        return 'Black'
    elif race_upper == 'WHITE' or 'WHITE' in race_upper:
        return 'White'
    elif 'ASIAN' in race_upper or 'PACIFIC' in race_upper or 'ISLANDER' in race_upper:
        return 'Asian/Pacific Islander'
    elif 'INDIAN' in race_upper or 'NATIVE' in race_upper or 'ALASKA' in race_upper:
        return 'American Indian/Alaska Native'
    elif 'UNKNOWN' in race_upper or 'OTHER' in race_upper:
        return 'Other'
    else:
        return 'Other'


gunshot_df['race_std'] = gunshot_df['race_ethnic_group'].apply(standardize_race)
print('\nRace distribution:')
print(gunshot_df['race_std'].value_counts())


# --- Age conversion ---
def convert_age(age):
    if pd.isna(age):
        return np.nan
    age_str = str(age).strip()
    if '_' in age_str or '-' in age_str:
        parts = age_str.replace('-', '_').split('_')
        try:
            return (int(parts[0]) + int(parts[1])) / 2
        except Exception:
            return np.nan
    try:
        return float(age_str)
    except Exception:
        return np.nan


gunshot_df['age_numeric'] = gunshot_df['age'].apply(convert_age)

def age_category(age):
    if pd.isna(age):
        return 'Unknown'
    elif age < 18:
        return '< 18'
    elif age < 25:
        return '18-24'
    elif age < 35:
        return '25-34'
    elif age < 45:
        return '35-44'
    elif age < 55:
        return '45-54'
    else:
        return '55+'

gunshot_df['age_cat'] = gunshot_df['age_numeric'].apply(age_category)


# --- Gender standardization ---
def standardize_gender(gender):
    if pd.isna(gender):
        return 'Unknown'
    gender_upper = str(gender).upper().strip()
    if gender_upper in ['MALE', 'M']:
        return 'Male'
    elif gender_upper in ['FEMALE', 'F']:
        return 'Female'
    else:
        return 'Unknown'


gunshot_df['gender_std'] = gunshot_df['gender'].apply(standardize_gender)


# --- Wound location standardization ---
# URSUS records received_force_location as a multi-select list of 11 body regions. The label
# spelling changed across eras (e.g., 'Rear upper torso/back' 2016-17, 'rear_upper_back'
# 2018-22, 'REAR_UPPER_TORSO' 2023-24), so each token is normalized to a region key and mapped
# by exact lookup. Substring matching is NOT safe here: a head/neck keyword such as 'EAR'
# would match inside 'REAR'.
REGION_TOKENS = {
    'HEAD': 'head', 'NECK/THROAT': 'neck', 'NECK_THROAT': 'neck',
    'FRONT UPPER TORSO/CHEST': 'front_upper', 'FRONT_UPPER_CHEST': 'front_upper',
    'FRONT_UPPER_TORSO': 'front_upper',
    'FRONT LOWER TORSO/ABDOMEN': 'front_lower', 'FRONT_LOWER_ABDOMEN': 'front_lower',
    'FRONT_LOWER_TORSO': 'front_lower',
    'FRONT BELOW WAIST/GROIN AREA': 'front_below_waist', 'FRONT_BELOW_WAIST': 'front_below_waist',
    'REAR UPPER TORSO/BACK': 'rear_upper', 'REAR_UPPER_BACK': 'rear_upper',
    'REAR_UPPER_TORSO': 'rear_upper',
    'REAR LOWER TORSO/BACK': 'rear_lower', 'REAR_LOWER_BACK': 'rear_lower',
    'REAR_LOWER_TORSO': 'rear_lower',
    'REAR BELOW WAIST/BUTTOCKS': 'rear_below_waist', 'REAR_BELOW_WAIST': 'rear_below_waist',
    'ARMS/HANDS': 'arms', 'ARMS_HANDS': 'arms', 'ARMS': 'arms',
    'FRONT LEGS/FEET': 'front_legs', 'FRONT_LEGS': 'front_legs',
    'REAR LEGS': 'rear_legs', 'REAR_LEGS': 'rear_legs',
}
NOT_APPLICABLE_TOKENS = {'(NOT APPLICABLE)', 'NOT_APPLICABLE', 'NOT APPLICABLE', ''}

# Primary analytic categories. Back = rear torso and buttocks, kept separate from the front
# torso because that is how URSUS records it; thorax_level_category() below provides the
# alternative grouping (upper torso front+rear = thorax) used as a sensitivity check.
REGION_TO_CATEGORY = {
    'head': 'Head/Neck', 'neck': 'Head/Neck',
    'front_upper': 'Chest',
    'front_lower': 'Abdomen', 'front_below_waist': 'Abdomen',
    'rear_upper': 'Back', 'rear_lower': 'Back', 'rear_below_waist': 'Back',
    'arms': 'Extremities', 'front_legs': 'Extremities', 'rear_legs': 'Extremities',
}
REGION_TO_THORAX_LEVEL = {
    'head': 'Head/Neck', 'neck': 'Head/Neck',
    'front_upper': 'Thorax', 'rear_upper': 'Thorax',
    'front_lower': 'Abdomen/Lower back', 'rear_lower': 'Abdomen/Lower back',
    'front_below_waist': 'Pelvis/Buttocks', 'rear_below_waist': 'Pelvis/Buttocks',
    'arms': 'Extremities', 'front_legs': 'Extremities', 'rear_legs': 'Extremities',
}
# Most-severe-region rule for assigning one primary category to multi-region cases
SEVERITY_ORDER = ['Head/Neck', 'Chest', 'Abdomen', 'Back', 'Extremities']
THORAX_LEVEL_ORDER = ['Head/Neck', 'Thorax', 'Abdomen/Lower back', 'Pelvis/Buttocks', 'Extremities']


def parse_regions(location):
    """Return the list of distinct URSUS region keys recorded for a civilian."""
    if pd.isna(location):
        return []
    regions = []
    for token in str(location).split(','):
        key = token.strip().upper()
        if key in NOT_APPLICABLE_TOKENS:
            continue
        if key not in REGION_TOKENS:
            raise ValueError(f'Unmapped URSUS wound-location token: {token!r}')
        if REGION_TOKENS[key] not in regions:
            regions.append(REGION_TOKENS[key])
    return regions


def primary_category(regions, mapping, order):
    cats = {mapping[r] for r in regions}
    for cat in order:
        if cat in cats:
            return cat
    return 'Unknown'


gunshot_df['wound_regions'] = gunshot_df['received_force_location'].apply(parse_regions)
gunshot_df['wound_location_std'] = gunshot_df['wound_regions'].apply(
    lambda r: primary_category(r, REGION_TO_CATEGORY, SEVERITY_ORDER))
gunshot_df['wound_location_thorax'] = gunshot_df['wound_regions'].apply(
    lambda r: primary_category(r, REGION_TO_THORAX_LEVEL, THORAX_LEVEL_ORDER))
print('\nWound location distribution:')
print(gunshot_df['wound_location_std'].value_counts())


# --- Wound count variables ---
# wound_count = number of distinct URSUS anatomical regions recorded as wounded (1-11). It is
# NOT a count of gunshot wounds: several bullets in one region count once.
SEVERITY_WEIGHTS = {'Head/Neck': 3, 'Chest': 3, 'Abdomen': 2, 'Back': 1, 'Extremities': 1}


def compute_wound_metrics(regions):
    if not regions:
        return pd.Series({'wound_count': np.nan, 'wound_count_high': np.nan,
                          'wound_count_low': np.nan, 'wound_severity_score': np.nan})
    cats = [REGION_TO_CATEGORY[r] for r in regions]
    high = sum(c in ('Head/Neck', 'Chest', 'Abdomen') for c in cats)
    return pd.Series({'wound_count': len(regions),
                      'wound_count_high': high,
                      'wound_count_low': len(regions) - high,
                      'wound_severity_score': sum(SEVERITY_WEIGHTS[c] for c in cats)})



wound_metrics = gunshot_df['wound_regions'].apply(compute_wound_metrics)
gunshot_df = pd.concat([gunshot_df, wound_metrics], axis=1)

print(f'\nWound count: mean={gunshot_df["wound_count"].mean():.1f}, '
      f'median={gunshot_df["wound_count"].median():.1f}, '
      f'range={int(gunshot_df["wound_count"].min())}-{int(gunshot_df["wound_count"].max())}')


# --- Resistance standardization ---
def categorize_resistance(resistance):
    if pd.isna(resistance):
        return 'Unknown'
    res_upper = str(resistance).upper()
    if any(x in res_upper for x in ['LIFE-THREATENING', 'LIFE_THREATENING', 'DEADLY', 'LETHAL']):
        return 'Armed/Deadly Force'
    if any(x in res_upper for x in ['WEAPON', 'FIREARM', 'GUN', 'KNIFE', 'ARMED']):
        return 'Armed/Deadly Force'
    if any(x in res_upper for x in ['ASSAULT', 'ATTACK', 'HIT', 'STRIKE', 'PUNCH', 'KICK', 'FIGHT']):
        return 'Physical Assault'
    if any(x in res_upper for x in ['ACTIVE', 'PHYSICAL', 'RESIST', 'PULL', 'STRUGGLE', 'TENSE']):
        return 'Physical Resistance'
    if any(x in res_upper for x in ['FLEE', 'FLIGHT', 'RUN', 'ESCAPE', 'EVADE', 'PURSUIT']):
        return 'Flight/Evasion'
    if any(x in res_upper for x in ['VERBAL', 'YELL', 'THREAT', 'REFUSE', 'NON-COMPLIANCE', 'NONCOMPLIANCE']):
        return 'Verbal Resistance'
    if any(x in res_upper for x in ['NONE', 'NO RESISTANCE', 'PASSIVE', 'COMPLIANT']):
        return 'None/Passive'
    return 'Other'


gunshot_df['resistance_std'] = gunshot_df['civilian_resistance_type'].apply(categorize_resistance)

# --- 2022 URSUS data bug fix: Confirmed_Armed contradicts Confirmed_Armed_Weapon ---
# In 2022 only, CA DOJ's public civilian_officer extract has civilian_confirmed_armed = False
# for records that nonetheless have a specific weapon recorded in
# civilian_confirmed_armed_weapon (verified against a freshly re-downloaded copy and against
# CA DOJ's own 2022 annual report; every other year 2016-2024 is internally consistent). Since
# civilian_resistance_type free text often doesn't independently mention the weapon, this also
# silently deflated the resistance_std/armed_assault classification for 2022 (48.6% -> 81.4%
# "armed" among gunshot cases after this fix, back in line with 69-87% in surrounding years).
# See docs/source_data_notes.md.
def has_informative_weapon(weapon):
    if pd.isna(weapon):
        return False
    weapon_str = str(weapon).strip().lower()
    return weapon_str not in ('', 'unknown', 'other')


confirmed_armed_bug_mask = (
    (gunshot_df['data_year'] == 2022)
    & gunshot_df['civilian_confirmed_armed'].astype(str).str.strip().str.upper().isin(['FALSE', 'NO', 'N', '0'])
    & gunshot_df['civilian_confirmed_armed_weapon'].apply(has_informative_weapon)
)
n_bug_fixed = int(confirmed_armed_bug_mask.sum())
print(f'\n2022 Confirmed_Armed/Confirmed_Armed_Weapon bug: reclassifying {n_bug_fixed} records '
      f'as armed (weapon field authoritative over contradicting boolean flag)')
gunshot_df.loc[confirmed_armed_bug_mask, 'resistance_std'] = 'Armed/Deadly Force'


# --- Armed status ---
def standardize_armed(row, armed_col, weapon_col):
    """Weapon field wins when it conflicts with a False armed flag (see bug fix above)."""
    if has_informative_weapon(row[weapon_col]):
        return 'Yes'
    armed = row[armed_col]
    if pd.isna(armed):
        return 'Unknown'
    armed_str = str(armed).upper().strip()
    if armed_str in ['YES', 'TRUE', 'Y', '1']:
        return 'Yes'
    elif armed_str in ['NO', 'FALSE', 'N', '0']:
        return 'No'
    else:
        return 'Unknown'


gunshot_df['perceived_armed_std'] = gunshot_df.apply(
    lambda r: standardize_armed(r, 'civilian_perceived_armed', 'civilian_perceived_armed_weapon'), axis=1)
gunshot_df['confirmed_armed_std'] = gunshot_df.apply(
    lambda r: standardize_armed(r, 'civilian_confirmed_armed', 'civilian_confirmed_armed_weapon'), axis=1)

# =============================================================================
# SECTION 4: MERGE INCIDENT DATA (COUNTY + OFFICER COUNTS)
# =============================================================================
print('\n' + '=' * 70)
print('SECTION 4: MERGING INCIDENT DATA')
print('=' * 70)

# CA DOJ changed incident ID column names across reporting eras:
#   2016-2021: incident_id
#   2022: incident_code
#   2023-2025: useofforceincidentid
# Create a unified key that works for all eras before merging.
def clean_id(val):
    if pd.isna(val):
        return None
    if isinstance(val, float):
        return str(int(val))
    return str(val)

gunshot_df['unified_incident_id'] = (
    gunshot_df['incident_id']
    .combine_first(gunshot_df.get('incident_code'))
    .combine_first(gunshot_df.get('useofforceincidentid'))
    .apply(clean_id)
)

n_unified = gunshot_df['unified_incident_id'].notna().sum()
print(f'Unified incident IDs created: {n_unified:,} of {len(gunshot_df):,}')

# Merge basic incident info (county, contact reason, officer counts)
incident_vars = ['incident_id', 'contact_reason', 'county', 'on_k12_campus',
                 'num_involved_officers', 'num_involved_civilians']
available_vars = [v for v in incident_vars if v in incidents.columns]
gunshot_df = gunshot_df.merge(
    incidents[available_vars],
    left_on='unified_incident_id', right_on='incident_id',
    how='left', suffixes=('_civ', '')
)

# 2023-24 (CIBRS) civilian-officer files name the agency column ORINumber (2025 is back to ORI); without this, ori is blank for every
# CIBRS case and agency clustering/FE silently drop those years.
gunshot_df['ori'] = gunshot_df['ori'].combine_first(gunshot_df.get('orinumber'))
gunshot_df['ori'] = gunshot_df['ori'].astype(str).str.strip().replace({'nan': np.nan, '': np.nan})
print(f'Records with ORI: {gunshot_df["ori"].notna().sum():,} of {len(gunshot_df):,}')

# --- Incident-circumstance measures from officer rows (all years) ---
# Officer rows record the force each officer RECEIVED, so these capture civilian threat
# behavior toward officers. URSUS has no per-officer "fired" flag (DISCHARGE_OF_FIREARM_
# INDIVIDUAL marks a recipient of gunfire), so officers_used_force is the closest available
# proxy for officers who acted: Officer_Used_Force (2016-22) or a force reason other than
# NO_FORCE (2023-25).
officers = pd.read_csv(DATA_DIR / 'master_officer_data_clean.csv', low_memory=False)
officers['unified_incident_id'] = (
    officers['incident_id']
    .combine_first(officers.get('incident_code'))
    .combine_first(officers.get('useofforceincidentid'))
    .apply(clean_id)
)
off_force = officers['received_force_type'].astype(str).str.upper()
# 2016 labels are plain English ("Discharge of firearm (miss)", "Threat of firearm"); other years
# use underscores. Match both, or 2016 civilians who fired at / threatened officers are missed.
officers['_civ_fired'] = off_force.str.contains(r'DISCHARGE[_ ]OF[_ ]FIREARM|DISCHARGE_FIREARM', regex=True)
officers['_threat_firearm'] = off_force.str.contains(r'THREAT[_ ]OF[_ ]FIREARM', regex=True)
# Vehicle used against the officer: 'Civilian vehicle contact' (2016), CIVILIAN_VEHICLE (2017-22),
# VEHICLE_CONTACT (2023, 2025), OFFICER_VEHICLE_CONTACT (2024, the only vehicle code that year).
# All are received-force codes, so each marks force the officer received (added 2026-10-12, PQ R2).
officers['_vehicle'] = off_force.str.contains('VEHICLE', regex=False)
officers['_injured'] =officers['injured'].astype(str).str.strip().str.upper().isin(['TRUE', 'YES'])
used_flag = officers['officer_officer_used_force'].astype(str).str.strip().str.upper()
reason = officers['officer_officer_used_force_reason'].astype(str).str.strip().str.upper()
officers['_used_force'] = np.where(
    officers['data_year'] <= 2022,
    used_flag.isin(['TRUE', 'YES']),
    ~reason.isin(['NO_FORCE', 'NAN', '']))
officer_agg = (officers.groupby('unified_incident_id')
               .agg(civ_fired_at_officer=('_civ_fired', 'max'),
                    civ_threat_firearm=('_threat_firearm', 'max'),
                    officer_injured=('_injured', 'max'),
                    civ_vehicle_against_officer=('_vehicle', 'max'),
                    officers_used_force=('_used_force', 'sum'))
               .reset_index())
for col in ['civ_fired_at_officer', 'civ_threat_firearm', 'officer_injured', 'civ_vehicle_against_officer']:
    officer_agg[col] = officer_agg[col].astype(int)
gunshot_df = gunshot_df.merge(officer_agg, on='unified_incident_id', how='left')


# --- Contact reason, harmonized across eras ---
# CIBRS (2023-25) has no "call for service" code; such contacts fall under
# CRIMINAL_SUSPICIOUS_ACTIVITY, so calls and crimes in progress are combined.
def harmonize_contact(reason):
    if pd.isna(reason):
        return 'Other/Unknown'
    r = str(reason).strip().upper()
    if any(x in r for x in ['CALL', 'IN_PROGRESS', 'CRIME IN PROGRESS', 'CRIMINAL_SUSPICIOUS']):
        return 'Call/crime in progress'
    if any(x in r for x in ['VEHICLE', 'TRAFFIC']):
        return 'Traffic/pedestrian stop'
    if any(x in r for x in ['PRE_PLANNED', 'PRE-PLANNED', 'WARRANT', 'COURT_ORDER', 'FOLLOWUP']):
        return 'Planned/warrant'
    return 'Other/Unknown'


gunshot_df['contact_reason_cat'] = gunshot_df['contact_reason'].apply(harmonize_contact)


# --- Confirmed weapon type (most dangerous recorded) ---
def weapon_type(weapon):
    if pd.isna(weapon):
        return 'None/unknown'
    tokens = {t.strip().replace(' ', '_') for t in str(weapon).upper().split(',')}
    if 'FIREARM' in tokens:
        return 'Firearm'
    if 'FIREARM_REPLICA' in tokens:
        return 'Replica firearm'
    # 2016-17 label 'Knife, blade, or stabbing instrument' splits into several tokens
    if any(t.startswith(('KNIFE', 'BLADE', 'OR_STABBING')) for t in tokens):
        return 'Knife/blade'
    if any('OTHER_DANGEROUS' in t for t in tokens):
        return 'Other weapon'
    return 'None/unknown'


gunshot_df['weapon_type'] = gunshot_df['civilian_confirmed_armed_weapon'].apply(weapon_type)
print('\nWeapon type:')
print(gunshot_df['weapon_type'].value_counts())
print('Contact reason:')
print(gunshot_df['contact_reason_cat'].value_counts())

# Merge agency names
if 'ORI' in agencies.columns:
    agency_names = agencies[['ORI', 'AGENCY_NAME']].drop_duplicates()
    agency_names.columns = ['ori', 'agency_name']
    gunshot_df = gunshot_df.merge(agency_names, on='ori', how='left')

# =============================================================================
# SECTION 5: VERIFY MERGE COVERAGE
# =============================================================================
print('\n' + '=' * 70)
print('SECTION 5: VERIFYING MERGE COVERAGE')
print('=' * 70)

# Use the canonical incident_id from the incidents table (merged above)
# and drop the civilian-era ID columns that are no longer needed
if 'incident_id_civ' in gunshot_df.columns:
    gunshot_df = gunshot_df.drop(columns=['incident_id_civ'], errors='ignore')

n_county = gunshot_df['county'].notna().sum()
n_officers = gunshot_df['num_involved_officers'].notna().sum()
print(f'Records with county: {n_county:,} of {len(gunshot_df):,} ({n_county/len(gunshot_df)*100:.1f}%)')
print(f'Records with officer count: {n_officers:,} of {len(gunshot_df):,} ({n_officers/len(gunshot_df)*100:.1f}%)')

if n_officers < len(gunshot_df) * 0.95:
    print(f'WARNING: {len(gunshot_df) - n_officers} records still missing officer count after merge.')

# =============================================================================
# SECTION 6: CREATE OFFICER COUNT CATEGORIES
# =============================================================================
print('\n' + '=' * 70)
print('SECTION 6: CREATING OFFICER COUNT VARIABLES')
print('=' * 70)

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

gunshot_df['officer_count_cat'] = gunshot_df['num_involved_officers'].apply(categorize_officer_count)
gunshot_df['multiple_officers'] = (gunshot_df['num_involved_officers'] >= 2).astype(float)
gunshot_df.loc[gunshot_df['num_involved_officers'].isna(), 'multiple_officers'] = np.nan

print('\nOfficer count distribution:')
print(f'  Mean: {gunshot_df["num_involved_officers"].mean():.2f}')
print(f'  SD: {gunshot_df["num_involved_officers"].std():.2f}')
print('\nOfficer count categories:')
print(gunshot_df['officer_count_cat'].value_counts())

print('\nCFR by officer count:')
for cat in ['1 officer', '2 officers', '3 officers', '4-5 officers', '6+ officers']:
    subset = gunshot_df[gunshot_df['officer_count_cat'] == cat]
    if len(subset) > 0:
        cfr = subset['fatal'].mean() * 100
        print(f'  {cat}: {subset["fatal"].sum()}/{len(subset)} = {cfr:.1f}%')

# =============================================================================
# SECTION 7: SAVE UNIFIED ANALYSIS-READY DATASET
# =============================================================================
print('\n' + '=' * 70)
print('SECTION 7: SAVING ANALYSIS DATASET')
print('=' * 70)

# Select and order final columns
analysis_columns = [
    'incident_id', 'ori', 'data_year',
    'fatal', 'injury_level',
    'race_std', 'age_numeric', 'age_cat', 'gender_std',
    'unified_incident_id',
    'wound_location_std', 'wound_location_thorax',
    'wound_count', 'wound_count_high', 'wound_count_low', 'wound_severity_score',
    'resistance_std',
    'perceived_armed_std', 'confirmed_armed_std', 'weapon_type',
    'num_involved_officers', 'officer_count_cat', 'multiple_officers', 'officers_used_force',
    'civ_fired_at_officer', 'civ_threat_firearm', 'officer_injured', 'civ_vehicle_against_officer', 'num_involved_civilians',
    'contact_reason', 'contact_reason_cat', 'county',
    'agency_name',
    'race_ethnic_group', 'age', 'gender',
    'received_force_location', 'civilian_resistance_type'
]

available_columns = [c for c in analysis_columns if c in gunshot_df.columns]
analysis_df = gunshot_df[available_columns].copy()

output_path = OUTPUT_DIR / 'california_analysis_ready.csv'
analysis_df.to_csv(output_path, index=False)
print(f'\nSaved: {output_path}')
print(f'  Total gunshot cases: {len(analysis_df):,}')
print(f'  Fatal: {analysis_df["fatal"].sum():,} ({analysis_df["fatal"].mean()*100:.1f}%)')
print(f'  Records with officer count: {analysis_df["num_involved_officers"].notna().sum():,}')
print(f'  Records with wound location: {(analysis_df["wound_location_std"] != "Unknown").sum():,}')

# Save CFR summary by individual officer count (matches TX format)
cfr_by_count = (analysis_df[analysis_df['num_involved_officers'].notna()]
    .groupby('num_involved_officers')
    .agg(fatal=('fatal', 'sum'), total=('fatal', 'count'))
    .reset_index())
cfr_by_count.columns = ['officers', 'fatal', 'total']
cfr_by_count['officers'] = cfr_by_count['officers'].astype(int)
cfr_by_count['cfr'] = cfr_by_count['fatal'] / cfr_by_count['total'] * 100
cfr_by_count.to_csv(TABLES_DIR / 'cfr_by_officer_count.csv', index=False)

print('\n' + '=' * 70)
print('DATA PREPARATION COMPLETE')
print('=' * 70)
print('\nDataset contains BOTH wound location AND officer count variables.')
print('Next step: Run 02_run_analysis.py for statistical models.')

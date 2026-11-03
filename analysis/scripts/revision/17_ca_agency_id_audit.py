"""
REVISION (PQ Round 3): California agency-identifier audit (Reviewer 2, data validation).

R2 asked how agency identifiers were harmonized across URSUS (2016-22, 2025) and CIBRS (2023-24)
and how many observations changed agencies. The round 2 fix (study2_california/01_prepare_data.py,
`ori.combine_first(orinumber)`) filled the agency ORI for CIBRS-era civilians, whose files name the
column `ORINumber` instead of `ORI`. This script documents:

  1. Which column carries the ORI in each year of the civilian master file, and whether any record
     carries both (if none does, the fix cannot have changed a non-missing ORI).
  2. ORI coverage among gunshot cases before and after the fix, by year.
  3. Agreement of each gunshot case's ORI with the ORI on its incident record (the incident table
     carries ORI in every year, so this is an independent check on the filled values).
  4. Continuity: whether CIBRS-era ORIs use the same identifiers as other years (format, overlap,
     and how many CIBRS-era cases belong to agencies also seen in other years).

  5. The third California correction (2022 armed-status flag): the number of 2022 gunshot cases
     whose Confirmed_Armed flag is False although a specific weapon is recorded, using the same
     rule as study2_california/01_prepare_data.py (weapon field authoritative).

Outputs (outputs/revision/):
  ca_agency_id_audit_by_year.csv  -- per-year coverage, fix effect, incident-ORI agreement
  ca_agency_id_audit_summary.csv  -- headline counts for the response memo
"""

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MERGED_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = MERGED_ROOT.parent / 'data' / 'processed'
OUTPUT_DIR = MERGED_ROOT / 'outputs' / 'revision'
CIBRS_YEARS = (2023, 2024)

civ = pd.read_csv(DATA_DIR / 'master_civilian_data_clean.csv', low_memory=False)
inc = pd.read_csv(DATA_DIR / 'master_incident_data_clean.csv', low_memory=False)
ready = pd.read_csv(MERGED_ROOT / 'outputs' / 'study2' / 'california_analysis_ready.csv')
reg = pd.read_csv(MERGED_ROOT / 'outputs' / 'study2' / 'california_regression_data.csv')


def clean(s):
    return s.astype(str).str.strip().str.upper().replace({'NAN': pd.NA, '': pd.NA})


# =============================================================================
# SECTION 1: SOURCE COLUMN BY YEAR (all civilians in the master file)
# =============================================================================
civ['ori_col'] = clean(civ['ori'])
civ['orinumber_col'] = clean(civ['orinumber'])
n_both = int((civ['ori_col'].notna() & civ['orinumber_col'].notna()).sum())

# =============================================================================
# SECTION 2-3: GUNSHOT CASES -- BEFORE/AFTER FIX, AND AGREEMENT WITH THE INCIDENT RECORD
# =============================================================================
ready['ori_after'] = clean(ready['ori'])
# Before the fix, CIBRS-era rows had no ORI; other years used the ORI column unchanged. This
# reconstruction makes ori_changed_where_present_before zero by construction; the substantive
# evidence that nothing was overwritten is civilian_rows_with_both_columns == 0.
ready['ori_before'] = ready['ori_after'].where(~ready['data_year'].isin(CIBRS_YEARS))
inc_ori = (inc.assign(incident_ori=clean(inc['ori']))
           [['incident_id', 'incident_ori']].drop_duplicates('incident_id'))
ready = ready.merge(inc_ori, left_on='unified_incident_id', right_on='incident_id',
                    how='left', suffixes=('', '_inc'))
ready['has_incident'] = ready['incident_ori'].notna()
ready['agrees_incident'] = ready['ori_after'] == ready['incident_ori']

by_year = []
for yr, d in ready.groupby('data_year'):
    c = civ[civ['data_year'] == yr]
    by_year.append({
        'year': yr,
        'era': 'CIBRS' if yr in CIBRS_YEARS else 'URSUS',
        'civilian_source_column': ('ORINumber' if c['orinumber_col'].notna().any() else 'ORI'),
        'gunshot_cases': len(d),
        'ori_before_fix': int(d['ori_before'].notna().sum()),
        'ori_after_fix': int(d['ori_after'].notna().sum()),
        'ori_changed_where_present_before': int(
            (d['ori_before'].notna() & (d['ori_before'] != d['ori_after'])).sum()),
        'with_incident_record': int(d['has_incident'].sum()),
        'agrees_with_incident_ori': int(d['agrees_incident'].sum()),
        'agencies': d['ori_after'].nunique(),
    })
by_year = pd.DataFrame(by_year)
by_year.to_csv(OUTPUT_DIR / 'ca_agency_id_audit_by_year.csv', index=False)

# =============================================================================
# SECTION 4: CONTINUITY OF IDENTIFIERS ACROSS ERAS
# =============================================================================
fmt_ok = ready['ori_after'].str.match(r'^CA[0-9A-Z]{7}$')
cibrs = ready[ready['data_year'].isin(CIBRS_YEARS)]
other = ready[~ready['data_year'].isin(CIBRS_YEARS)]
cibrs_oris = set(cibrs['ori_after'].dropna())
other_oris = set(other['ori_after'].dropna())

# =============================================================================
# SECTION 5: 2022 ARMED-STATUS FLAG (same rule as 01_prepare_data.py)
# =============================================================================
gun = civ[civ['received_force_type'].str.upper().fillna('').str.contains('FIREARM.*HIT|HIT.*FIREARM', regex=True)]
weapon = gun['civilian_confirmed_armed_weapon'].astype(str).str.strip().str.lower()
informative = gun['civilian_confirmed_armed_weapon'].notna() & ~weapon.isin(['', 'unknown', 'other'])
flag_false = gun['civilian_confirmed_armed'].astype(str).str.strip().str.upper().isin(['FALSE', 'NO', 'N', '0'])
contradict = flag_false & informative
armed_by_year = contradict.groupby(gun['data_year']).sum()

reg_cibrs = reg[reg['data_year'].isin(CIBRS_YEARS)]
summary = pd.DataFrame([{
    'civilian_rows_with_both_columns': n_both,
    'gunshot_cases': len(ready),
    'cibrs_gunshot_cases': len(cibrs),
    'cibrs_cases_ori_filled': int(cibrs['ori_after'].notna().sum()),
    'cibrs_regression_cases': len(reg_cibrs),
    'ori_changed_where_present_before': int(by_year['ori_changed_where_present_before'].sum()),
    'ori_missing_after_fix': int(ready['ori_after'].isna().sum()),
    # One 2020 case has no incident record (and no officer count; not in the regression sample)
    'cases_without_incident_record': int((~ready['has_incident']).sum()),
    'agrees_with_incident_ori_pct': round(ready.loc[ready['has_incident'], 'agrees_incident'].mean() * 100, 2),
    'disagree_with_incident_ori': int((ready['has_incident'] & ~ready['agrees_incident']).sum()),
    'ori_format_ok_pct': round(fmt_ok.mean() * 100, 2),
    'agencies_total': ready['ori_after'].nunique(),
    'agencies_cibrs': len(cibrs_oris),
    'agencies_cibrs_also_other_years': len(cibrs_oris & other_oris),
    'agencies_cibrs_only': len(cibrs_oris - other_oris),
    'cibrs_cases_agency_seen_other_years': int(cibrs['ori_after'].isin(other_oris).sum()),
    'gunshot_2022': int((gun['data_year'] == 2022).sum()),
    'armed_flag_fixed_2022': int(armed_by_year.get(2022, 0)),
    'armed_flag_contradictions_other_years': int(armed_by_year.drop(2022, errors='ignore').sum()),
}])
summary.to_csv(OUTPUT_DIR / 'ca_agency_id_audit_summary.csv', index=False)

print(by_year.to_string(index=False))
print(summary.T.to_string())

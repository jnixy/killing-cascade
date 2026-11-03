"""
Study 1 - Texas: Build the analysis input file, 2015-2025, from TJI + OAG reports.

Run after 00_build_oag_extension.py and 00b_validate_and_collapse_oag.py.

What it does to the TJI data (decided 2026-10-02):
  1. Officer counts. TJI's num_officers_recorded adds officers across a person's reports, but
     in multi-agency shootings each agency often lists every officer who fired (e.g., El Paso
     2022-06-29: two reports listing the same 3 officers; TJI = 6). Where the OAG reports for
     the person are available, the count is replaced by the number of DISTINCT officers across
     those reports. A replacement is made only when the OAG reports contain at least as many
     officer listings as TJI recorded (num_officers_summed >= TJI count); otherwise the portal
     may be missing a report TJI saw, and TJI's count is kept and flagged.
  2. Duplicate rows. TJI sometimes gives one person a row per agency report (e.g., Donley
     2024-05-10). TJI rows that link to the same OAG person are merged into one row.
  3. Extension. People shot 2024-08-27..2025-12-31 come from the OAG reports
     (data/staging_2025/tx_oag_extension_2024_2025.csv).
  4. Coverage gaps. OAG-reported people shot on or before TJI's last date who have no TJI row
     are added (source = 'OAG (missing from TJI)'), unless an unmatched TJI row lies within
     2 days in the same county (then it is likely the same person recorded differently).

Linking TJI rows to OAG persons:
  - 2020+: by report number (TJI lists them, joined with "/").
  - before 2020: TJI has no report numbers, so rows are matched on incident date (+/- 1 day),
    county, and the person's sex, race, and age (+/- 1). Only unique matches are used. The
    same attribute rule is scored against the report-number links for 2020+ (printed below).
  - 2015: the portal holds no reports, so TJI is used as-is.

Output: data/processed/texas_ois_2015_2025.csv (TJI schema + provenance columns)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
STAGING_DIR = PROJECT_ROOT / 'data' / 'staging_2025'
OUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'texas_ois_2015_2025.csv'
TJI_PATH = PROJECT_ROOT / 'archive' / 'officer_count_paper' / 'Study 1 - Texas' / 'data' / 'TJI data raw.csv'
REPORT_NO_LINK_START = pd.Timestamp('2020-01-01')

# =============================================================================
# SECTION 1: LOAD
# =============================================================================
tji = pd.read_csv(TJI_PATH, low_memory=False)
tji['date_incident'] = pd.to_datetime(tji['date_incident'])
tji['tji_row'] = np.arange(len(tji))
persons = pd.read_csv(STAGING_DIR / 'tx_oag_persons_all.csv', parse_dates=['date_incident'])
ext = pd.read_csv(STAGING_DIR / 'tx_oag_extension_2024_2025.csv', parse_dates=['date_incident'])
print(f'TJI rows: {len(tji):,}; OAG persons (all years): {len(persons):,}; extension: {len(ext):,}')

# =============================================================================
# SECTION 2: LINK TJI ROWS TO OAG PERSONS
# =============================================================================
# 2a. By report number (2020+)
rep2person = {r: k for k, nos in zip(persons['person_key'], persons['all_report_nos'])
              for r in nos.split('/')}


def link_by_number(s):
    keys = []
    for part in str(s).split('/'):
        d = re.sub(r'\D', '', part)
        if d and f'OIS-{d.zfill(8)}' in rep2person:
            keys.append(rep2person[f'OIS-{d.zfill(8)}'])
    return pd.Series(keys).mode().iloc[0] if keys else np.nan


tji['person_by_number'] = tji['ois_report_no'].map(link_by_number)

# 2b. By attributes (used before 2020; scored on 2020+)
p = persons.assign(county=persons['incident_county'].str.upper().str.strip())


def link_by_attributes(r):
    c = p[(p['county'] == str(r['incident_county']).upper().strip())
          & ((p['date_incident'] - r['date_incident']).abs() <= pd.Timedelta('1D'))
          & (p['civilian_gender'] == r['civilian_gender'])
          & (p['civilian_race'] == r['civilian_race'])]
    if pd.notna(r['civilian_age']):
        c = c[c['civilian_age'].isna() | ((c['civilian_age'] - r['civilian_age']).abs() <= 1)]
    return c['person_key'].iloc[0] if len(c) == 1 else np.nan


tji['person_by_attr'] = tji.apply(link_by_attributes, axis=1)
recent = tji[tji['date_incident'] >= REPORT_NO_LINK_START]
both = recent.dropna(subset=['person_by_attr'])
print(f'\nAttribute linking scored on 2020+ rows: matched {len(both):,} of {len(recent):,} '
      f'({len(both) / len(recent):.1%}); agrees with report-number link in '
      f'{(both["person_by_attr"] == both["person_by_number"]).mean():.2%}')

tji['person_key'] = np.where(tji['date_incident'] >= REPORT_NO_LINK_START,
                             tji['person_by_number'], tji['person_by_attr'])
tji['link_method'] = np.where(tji['person_key'].isna(), 'none',
                              np.where(tji['date_incident'] >= REPORT_NO_LINK_START, 'report_no', 'attributes'))
print('\nTJI rows linked to an OAG person, by year:')
print(pd.crosstab(tji['date_incident'].dt.year, tji['link_method']).to_string())

# =============================================================================
# SECTION 3: MERGE DUPLICATE TJI ROWS (same OAG person)
# =============================================================================
tji = tji.merge(persons[['person_key', 'num_officers_recorded', 'num_officers_summed', 'all_report_nos']]
                .rename(columns={'num_officers_recorded': 'oag_officers_distinct',
                                 'num_officers_summed': 'oag_officers_summed',
                                 'all_report_nos': 'oag_report_nos'}),
                on='person_key', how='left')
tji['tji_officer_count'] = tji['num_officers_recorded']
grp = tji['person_key'].fillna('tji_' + tji['tji_row'].astype(str))
tji['n_tji_rows_merged'] = grp.map(grp.value_counts())
dups = tji[tji['n_tji_rows_merged'] > 1]
print(f'\nDuplicate TJI rows: {len(dups)} rows describing {dups["person_key"].nunique()} people')

merged = []
for key, g in tji.groupby(grp, sort=False):
    # Lead row = the one with the largest officer count (usually the primary agency's report).
    row = g.sort_values(['tji_officer_count', 'tji_row'], ascending=[False, True]).iloc[0].copy()
    if len(g) > 1:
        row['tji_officer_count'] = g['tji_officer_count'].sum()        # TJI's rule across its rows
        row['civilian_died'] = 'DEATH' if (g['civilian_died'] == 'DEATH').any() else 'INJURY'
        row['deadly_weapon'] = g['deadly_weapon'].max()
        row['multiple_officers_involved'] = g['multiple_officers_involved'].max()
        # Each extra TJI row was counted as another person in TJI's per-incident row count.
        row['num_rows_about_this_incident'] = max(1, row['num_rows_about_this_incident'] - (len(g) - 1))
        row['ois_report_no'] = ' / '.join(g['ois_report_no'].dropna().astype(str))
    merged.append(row)
tx = pd.DataFrame(merged).reset_index(drop=True)

# =============================================================================
# SECTION 4: OFFICER COUNTS
# =============================================================================
verifiable = tx['oag_officers_summed'].notna() & (tx['oag_officers_summed'] >= tx['tji_officer_count'])
tx['num_officers_recorded'] = np.where(verifiable, tx['oag_officers_distinct'], tx['tji_officer_count']).astype(int)
tx['officer_count_source'] = np.select(
    [verifiable, tx['person_key'].notna()],
    ['OAG distinct officers', 'TJI (OAG reports incomplete)'], 'TJI (no OAG report)')
changed = tx['num_officers_recorded'] != tx['tji_officer_count']
print('\nOfficer-count source by year:')
print(pd.crosstab(tx['date_incident'].dt.year, tx['officer_count_source']).to_string())
print(f'\nCounts changed: {changed.sum()} rows '
      f'(lower: {(tx["num_officers_recorded"] < tx["tji_officer_count"]).sum()}, '
      f'higher: {(tx["num_officers_recorded"] > tx["tji_officer_count"]).sum()}); '
      f'2+ -> 1: {((tx["tji_officer_count"] >= 2) & (tx["num_officers_recorded"] == 1)).sum()}')
print(pd.crosstab(tx['tji_officer_count'].clip(upper=6), tx['num_officers_recorded'].clip(upper=6),
                  rownames=['TJI'], colnames=['corrected']).to_string())
# Officer count in the data is now "distinct officers who fired"; the TJI flag that says
# whether more than one officer was involved stays as reported.
tx['source'] = 'TJI'

# =============================================================================
# SECTION 5: APPEND THE OAG EXTENSION
# =============================================================================
def as_oag_rows(d, source):
    d = d.rename(columns={'num_officers_recorded': 'oag_officers_distinct',
                          'num_officers_summed': 'oag_officers_summed'}).copy()
    d['num_officers_recorded'] = d['oag_officers_distinct'].astype(int)
    d['officer_count_source'] = 'OAG distinct officers'
    d['link_method'] = 'OAG source'
    d['n_tji_rows_merged'] = 0
    d['civilian_died'] = d['civilian_died'].str.upper()
    d['weapon_reported_by_media_category'] = '(DETAILS MISSING)'   # TJI media coding not in OAG
    d['source'] = source
    return d


ext = as_oag_rows(ext, 'OAG')

# =============================================================================
# SECTION 5b: ADD OAG-REPORTED PEOPLE MISSING FROM TJI (decided 2026-10-02)
# =============================================================================
# People in the OAG reports for incidents on or before TJI's last date who match no TJI row.
# Before 2020 TJI rows are matched by attributes, so an unmatched OAG person may simply be a
# TJI row whose race or age was recorded differently; those are left out when an unmatched TJI
# row exists within 2 days in the same county (likely the same person).
TJI_END = tji['date_incident'].max()
unmatched_tji = tx[tx['person_key'].isna()].assign(county=lambda d: d['incident_county'].str.upper().str.strip())


def has_unmatched_tji_nearby(r):
    return ((unmatched_tji['county'] == str(r['incident_county']).upper().strip())
            & ((unmatched_tji['date_incident'] - r['date_incident']).abs() <= pd.Timedelta('2D'))).any()


cand = persons[(persons['date_incident'] <= TJI_END) & ~persons['person_key'].isin(set(tx['person_key'].dropna()))]
near = cand.apply(has_unmatched_tji_nearby, axis=1)
gap = as_oag_rows(cand[~near], 'OAG (missing from TJI)')
print(f'\nOAG people with no TJI row (through {TJI_END.date()}): {len(cand)}; '
      f'likely already in TJI under different race/age: {near.sum()}; added: {len(gap)}')
print(gap.groupby(gap['date_incident'].dt.year).size().to_string())

out = pd.concat([tx] + [d[[c for c in tx.columns if c in d.columns]] for d in (ext, gap)], ignore_index=True)
out = out.sort_values('date_incident').reset_index(drop=True)
out.to_csv(OUT_PATH, index=False)

# Injury-of-Officer reports (date + county), used by 01_prepare_data.py to flag shootings in
# which the civilian shot an officer. Re-parsed from the current portal, replacing the old
# tx_ois_reports_detailed.csv, which stopped at Feb 2026 and came from the old parser.
parsed = pd.read_csv(STAGING_DIR / 'tx_oag_reports_parsed.csv', low_memory=False, parse_dates=['incident_dt'])
inj = parsed.loc[parsed['Report_Type__c'] == 'Injury of Officer', ['report_no', 'incident_dt', 'incident_county']]
inj = inj.rename(columns={'incident_dt': 'Date_of_Incident', 'incident_county': 'Incident_County'})
inj['Report_Type'] = 'Injury of Officer'
inj.to_csv(OUT_PATH.parent / 'tx_oag_injury_of_officer.csv', index=False)
print(f'Injury-of-Officer reports: {len(inj):,} (unparsed date {inj["Date_of_Incident"].isna().sum()}, '
      f'county {inj["Incident_County"].isna().sum()})')

# Data-construction facts cited in the manuscript's Methods (read inline there).
fields = pd.read_csv(STAGING_DIR / 'tx_oag_validation_fields.csv')
person_fields = ['date_incident', 'incident_county', 'civilian_gender', 'civilian_age', 'civilian_race',
                 'civilian_died', 'deadly_weapon', 'incident_type_cat (analysis coding)']
pf = fields[fields['field'].isin(person_fields)]
summary = {
    'tji_rows_raw': len(tji),
    'tji_last_incident': tji['date_incident'].max().date().isoformat(),
    'tji_duplicate_rows': len(dups),
    'tji_duplicate_people': dups['person_key'].nunique(),
    'tji_rows_after_merge': len(tx),
    'oag_extension_people': len(ext),
    'oag_missing_from_tji_candidates': len(cand),
    'oag_missing_from_tji_likely_dup': int(near.sum()),
    'oag_missing_from_tji_added': len(gap),
    'oag_reports_parsed': int(parsed['Report_Type__c'].eq('Officer Involved').sum()),
    'validation_tji_rows': int(pf['n'].max()),
    'validation_min_pct': pf['pct_agree'].min(),
    'validation_max_pct': pf['pct_agree'].max(),
    'validation_tji_count_eq_listing_pct': fields.loc[fields['field'].str.startswith('num_officers, TJI rule'),
                                                      'pct_agree'].iloc[0],
    'attr_link_agree_pct': round(100 * (both['person_by_attr'] == both['person_by_number']).mean(), 2),
    'counts_verified': int((tx['officer_count_source'] == 'OAG distinct officers').sum()),
    'counts_changed': int(changed.sum()),
    'counts_lower': int((tx['num_officers_recorded'] < tx['tji_officer_count']).sum()),
    'counts_higher': int((tx['num_officers_recorded'] > tx['tji_officer_count']).sum()),
}
tables_dir = PROJECT_ROOT / 'analysis' / 'outputs' / 'study1' / 'tables'
tables_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame([summary]).to_csv(tables_dir / 'tx_data_construction.csv', index=False)
print('\nData-construction summary:', summary)

print(f'\nSaved {OUT_PATH.relative_to(PROJECT_ROOT)}: {len(out):,} rows '
      f'(TJI {len(tx):,} after merging {len(tji) - len(tx)} duplicate rows; OAG extension {len(ext):,}; '
      f'OAG missing from TJI {len(gap):,})')
print(out.groupby(out['date_incident'].dt.year).agg(
    n=('source', 'size'), fatal=('civilian_died', lambda s: (s == 'DEATH').sum()),
    multi_officer=('num_officers_recorded', lambda s: (s >= 2).mean().round(3))).to_string())

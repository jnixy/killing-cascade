"""
Study 1 - Texas: Validate the OAG PDF parser against TJI, then build the post-TJI extension
(Phase 1, staging only).

Input:  data/staging_2025/tx_oag_reports_parsed.csv   (from 00_build_oag_extension.py)
        archive/.../TJI data raw.csv
Output: data/staging_2025/tx_oag_validation_fields.csv      field agreement with TJI
        data/staging_2025/tx_oag_validation_mismatches.csv  every disagreement, for hand-checking
        data/staging_2025/tx_oag_coverage_by_quarter.csv    persons per quarter, OAG vs TJI
        data/staging_2025/tx_oag_persons_all.csv            every OAG person since 2020 (audit trail)
        data/staging_2025/tx_oag_extension_2024_2025.csv    TJI-schema rows, 2024-08-27..2025-12-31

Unit of analysis is the person shot (TJI's row). The OAG reports do not map one-to-one:
  - each agency whose officers fired files its own report on the person;
  - some agencies file one report per officer (e.g., several DPS reports on one person);
  - an amendment can be filed as a NEW report number while the original stays "current"
    (e.g., OIS-00002632 amended / OIS-00002634 original), and some reports are re-filed
    verbatim (e.g., OIS-00004714 / OIS-00004716).
So reports are grouped into persons by attributes, and officers are de-duplicated across the
person's reports rather than summed.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
STAGING_DIR = PROJECT_ROOT / 'data' / 'staging_2025'
TJI_PATH = PROJECT_ROOT / 'archive' / 'officer_count_paper' / 'Study 1 - Texas' / 'data' / 'TJI data raw.csv'

VALIDATION_START = pd.Timestamp('2020-01-01')
TJI_END = pd.Timestamp('2024-08-26')
STUDY_END = pd.Timestamp('2025-12-31')
SAME_PERSON_WINDOW = pd.Timedelta('24h')   # agencies' reported times for one shooting differ by up to ~6h
CROSS_COUNTY_WINDOW = pd.Timedelta('2h')
MAX_OFFICERS = 15

# =============================================================================
# SECTION 1: MAP OAG VALUES TO TJI CODES
# =============================================================================

RACE = {'ANGLO OR WHITE': 'WHITE', 'WHITE': 'WHITE',
        'BLACK OR AFRICAN AMERICAN': 'BLACK', 'BLACK': 'BLACK',
        'HISPANIC OR LATINO': 'HISPANIC', 'HISPANIC': 'HISPANIC'}
# Everything else that parsed (Asian, American Indian, Other, ...) is TJI's OTHER; a value
# that did not parse stays missing.


def map_race(x):
    if pd.isna(x) or not str(x).strip():
        return np.nan
    return RACE.get(re.sub(r'\s+', ' ', str(x)).strip().upper(), 'OTHER')


def map_weapon(x):
    """OAG item 7. Returns True/False, or NaN if the text is not a recognized answer."""
    if pd.isna(x):
        return np.nan
    s = str(x).upper()
    if 'DID NOT' in s:
        return False
    if 'CARRIED, EXHIBITED, OR USED A DEADLY WEAPON' in s:
        return True
    return np.nan


def map_yes_no(x):
    s = str(x).strip().upper()
    return True if s == 'YES' else (False if s == 'NO' else np.nan)


# The PDF prints the "Specify type of call:" free-text box after the selected options
# (whether or not "Other" was picked). TJI (2019+) keeps only the selected options, with
# "Other" as plain "OTHER", so the free text is dropped to match.
def normalize_call(s):
    if pd.isna(s):
        return s
    s = re.sub(r'\s*-?\s*SPECIFY TYPE OF CALL\b.*$', '', str(s))
    return re.sub(r'\s*;\s*', '; ', s).strip()


def categorize_incident(x):
    """Copied verbatim from study1_texas/01_prepare_data.py so validation uses the analysis coding."""
    if pd.isna(x):
        return 'Unknown'
    x = str(x).lower()
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


def to_tji_schema(p):
    """Report-level OAG fields in TJI's codes."""
    out = pd.DataFrame({'report_no': p['report_no'], 'version_type': p['Version_Type__c']})
    out['incident_dt'] = p['incident_dt']
    out['date_incident'] = p['incident_dt'].dt.normalize()
    out['time_incident'] = p['incident_dt'].dt.strftime('%H:%M:%S')
    out['incident_county'] = p['incident_county'].str.upper().str.strip()
    out['incident_city'] = p['incident_city'].str.upper().str.strip()
    out['street_no'] = p['incident_street'].str.extract(r'(\d+)', expand=False)
    out['civilian_gender'] = p['civ_gender'].str.upper().str.strip()
    out['civilian_age'] = pd.to_numeric(p['civ_age'], errors='coerce')
    out['civilian_race'] = p['civ_race'].map(map_race)
    out['civilian_died'] = p['incident_result'].str.upper().str.strip().map(
        {'DEATH': 'DEATH', 'INJURY': 'INJURY'})
    out['deadly_weapon'] = p['civ_weapon'].map(map_weapon)
    out['multiple_officers_involved'] = p['multiple_officers_response'].map(map_yes_no)
    out['incident_result_of'] = p['incident_result_of'].str.upper().str.strip().map(normalize_call)
    out['agency_name'] = p['agency_name'].str.upper().str.strip()
    out['n_officers'] = p['n_officer_rows']
    out['officer_rows_ok'] = p['n_officer_rows'] == p['n_duty_tokens']
    for i in range(1, MAX_OFFICERS + 1):
        if f'officer_gender_{i}' in p:
            out[f'officer_gender_{i}'] = p[f'officer_gender_{i}'].str.upper()
            out[f'officer_age_{i}'] = pd.to_numeric(p[f'officer_age_{i}'], errors='coerce')
            out[f'officer_race_{i}'] = p[f'officer_race_{i}'].map(map_race)
            out[f'officer_on_duty_{i}'] = p[f'officer_duty_{i}'].str.upper().eq('ON DUTY').where(
                p[f'officer_duty_{i}'].notna())
    return out


def officers_of(r):
    """(sex, race, age) of each officer listed on one report."""
    return [(r.get(f'officer_gender_{i}'), r.get(f'officer_race_{i}'), r.get(f'officer_age_{i}'))
            for i in range(1, MAX_OFFICERS + 1) if not pd.isna(r.get(f'officer_gender_{i}'))]


# =============================================================================
# SECTION 2: GROUP REPORTS INTO PERSONS
# =============================================================================


def assign_person_ids(rep):
    """Union reports that describe the same person.

    Agency is ignored: the same person is reported by every agency that fired, and some
    agencies file one report per officer. Two reports within SAME_PERSON_WINDOW of each other
    with the same sex are the same person if either
      (strict)  same county and race, ages within 1 year (a missing age matches any); or
      (relaxed) same exact age, and either same county (agencies sometimes record race
                differently: "Not Available", "Other"), or same race with incident times
                within CROSS_COUNTY_WINDOW (an agency may name the neighboring county, e.g.,
                Uvalde/Zavala 2022-05-24, Harris/Fort Bend 2022-07-23). Without the time limit,
                same-age shootings in different cities on the same day were merged (Austin/
                Houston 2021-01-05, 10 hours apart).
    and in every case
      - reports more than CROSS_COUNTY_WINDOW apart must share the street number (or, if
        either street is blank, be within 6 hours) -- e.g., Midland 2023-03-25/26 and
        Harris 2020-04-21/22 are different shootings with near-identical people;
      - two reports from the SAME agency with different known ages are different people
        (an agency files one report per person, and a verbatim re-filing keeps the age;
        e.g., Universal City 2020-04-01, ages 16 and 17).
    """
    rep = rep.reset_index(drop=True)
    parent = list(range(len(rep)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for _, g in rep.groupby('civilian_gender', dropna=False):
        g = g.sort_values('incident_dt')
        idx = g.index.to_list()
        for a_pos, a in enumerate(idx):
            for b in idx[a_pos + 1:]:
                if rep.at[b, 'incident_dt'] - rep.at[a, 'incident_dt'] > SAME_PERSON_WINDOW:
                    break
                same_county = rep.at[a, 'incident_county'] == rep.at[b, 'incident_county']
                same_race = rep.at[a, 'civilian_race'] == rep.at[b, 'civilian_race']
                age_a, age_b = rep.at[a, 'civilian_age'], rep.at[b, 'civilian_age']
                strict = same_county and same_race and (
                    pd.isna(age_a) or pd.isna(age_b) or abs(age_a - age_b) <= 1)
                gap = rep.at[b, 'incident_dt'] - rep.at[a, 'incident_dt']
                close = gap <= CROSS_COUNTY_WINDOW
                relaxed = (not pd.isna(age_a)) and age_a == age_b and (
                    same_county or (same_race and close))
                st_a, st_b = rep.at[a, 'street_no'], rep.at[b, 'street_no']
                place_ok = close or (st_a == st_b if not (pd.isna(st_a) or pd.isna(st_b))
                                     else gap <= pd.Timedelta('6h'))
                same_agency_diff_age = (rep.at[a, 'agency_name'] == rep.at[b, 'agency_name']
                                        and not pd.isna(age_a) and not pd.isna(age_b) and age_a != age_b)
                if (strict or relaxed) and place_ok and not same_agency_diff_age:
                    parent[find(b)] = find(a)
    roots = [find(i) for i in range(len(rep))]
    first_no = rep.groupby(pd.Series(roots))['report_no'].transform('min')
    return rep.assign(person_key=first_no.values)


def collapse_persons(rep):
    """One TJI-schema row per person."""
    rows = []
    for key, g in rep.sort_values('report_no').groupby('person_key'):
        # Prefer an amended report for the person's own fields, else the earliest filing.
        lead = g[g['version_type'] == 'AMENDED'].iloc[0] if (g['version_type'] == 'AMENDED').any() else g.iloc[0]
        # Officers de-duplicated across the person's reports: a (sex, race, age) officer is
        # counted as many times as the most it appears on any single report.
        tally = {}
        for _, r in g.iterrows():
            counts = pd.Series([str(o) for o in officers_of(r)]).value_counts()
            for o, c in counts.items():
                tally[o] = max(tally.get(o, 0), c)
        lead_offs = officers_of(lead)
        rows.append({
            'person_key': key,
            'ois_report_no': lead['report_no'],
            'all_report_nos': '/'.join(g['report_no']),
            'date_incident': g['date_incident'].min(),
            'time_incident': lead['time_incident'],
            'incident_county': lead['incident_county'],
            'incident_city': lead['incident_city'],
            'civilian_gender': lead['civilian_gender'],
            'civilian_age': lead['civilian_age'],
            'civilian_race': lead['civilian_race'],
            # A later report can record a death an earlier one did not, so death on any counts.
            'civilian_died': 'DEATH' if (g['civilian_died'] == 'DEATH').any() else 'INJURY',
            'died_conflict': g['civilian_died'].nunique() > 1,
            'deadly_weapon': lead['deadly_weapon'],
            'weapon_conflict': g['deadly_weapon'].nunique() > 1,
            'multiple_officers_involved': g['multiple_officers_involved'].max(),
            'incident_result_of': lead['incident_result_of'],
            'agency_name_1': lead['agency_name'],
            'n_agencies': g['agency_name'].nunique(),
            'officer_gender_1': lead_offs[0][0] if lead_offs else np.nan,
            'officer_race_1': lead_offs[0][1] if lead_offs else np.nan,
            'officer_age_1': lead_offs[0][2] if lead_offs else np.nan,
            'officer_on_duty': lead.get('officer_on_duty_1'),
            'num_officers_recorded': sum(tally.values()),
            'num_officers_summed': int(g['n_officers'].sum()),
            'num_reports_filed': len(g),
        })
    persons = pd.DataFrame(rows)
    # TJI's "rows about this incident": people shot in the same incident, here the same date,
    # time, and county. (Tested against TJI on the overlap: this agreed more often than
    # date + county + agency, which splits multi-agency incidents and merges same-day ones.)
    inc_key = (persons['date_incident'].astype(str) + ' ' + persons['time_incident'].fillna('?')
               + '|' + persons['incident_county'].fillna('?'))
    persons['num_rows_about_this_incident'] = inc_key.map(inc_key.value_counts())
    return persons


# =============================================================================
# SECTION 3: MAIN
# =============================================================================

if __name__ == '__main__':
    parsed = pd.read_csv(STAGING_DIR / 'tx_oag_reports_parsed.csv', low_memory=False,
                         parse_dates=['incident_dt', 'index_incident_date'])
    oi = parsed[parsed['Report_Type__c'] == 'Officer Involved'].copy()
    rep = assign_person_ids(to_tji_schema(oi))
    persons = collapse_persons(rep)

    # ---- 3a. Parse completeness ----
    core = ['date_incident', 'incident_county', 'civilian_gender', 'civilian_age', 'civilian_race',
            'civilian_died', 'deadly_weapon', 'incident_result_of', 'agency_name']
    print(f'Officer Involved reports parsed (incidents since {VALIDATION_START.date()}): {len(rep):,}')
    print('Unparsed rate by field:')
    print(rep[core].isna().mean().round(4).to_string())
    print(f'Officer rows == duty tokens: {rep["officer_rows_ok"].mean():.4f}; '
          f'reports with zero officer rows: {(rep["n_officers"] == 0).sum()}')
    print('Raw item-7 values:', oi['civ_weapon'].value_counts(dropna=False).to_dict())
    print(f'Persons: {len(persons):,} from {len(rep):,} reports')

    # ---- 3b. Link TJI rows to OAG reports (TJI joins several report numbers with "/") ----
    tji = pd.read_csv(TJI_PATH, low_memory=False)
    tji['date_incident'] = pd.to_datetime(tji['date_incident'])
    tji = tji[tji['date_incident'] >= VALIDATION_START].reset_index(drop=True)
    tji['tji_row'] = tji.index
    nums = (tji['ois_report_no'].fillna('').str.split('/').explode().str.strip()
            .map(lambda s: 'OIS-' + re.sub(r'\D', '', s).zfill(8) if re.sub(r'\D', '', s) else np.nan))
    links = pd.DataFrame({'tji_row': nums.index, 'report_no': nums.values}).dropna()
    links = links.merge(rep[['report_no', 'person_key']], on='report_no')
    tji_person = links.groupby('tji_row')['person_key'].agg(lambda s: s.mode().iloc[0])
    tji['person_key'] = tji['tji_row'].map(tji_person)
    tji['first_report'] = tji['tji_row'].map(links.groupby('tji_row')['report_no'].first())
    print(f'\nTJI rows since {VALIDATION_START.date()}: {len(tji):,}; '
          f'linked to an OAG report: {tji["person_key"].notna().sum():,}')
    rows_per_person = tji['person_key'].value_counts()
    print(f'OAG persons that TJI splits across 2+ rows: {(rows_per_person > 1).sum()} '
          f'(covering {rows_per_person[rows_per_person > 1].sum()} TJI rows)')

    # ---- 3c. Report-level field agreement (TJI row vs its first linked OAG report) ----
    m = tji.merge(rep, left_on='first_report', right_on='report_no', suffixes=('_tji', '_oag'))

    def agree(a, b):
        return (a == b) | (a.isna() & b.isna())

    checks = {
        'date_incident': agree(m['date_incident_tji'], m['date_incident_oag']),
        'incident_county': agree(m['incident_county_tji'].str.upper().str.strip(), m['incident_county_oag']),
        'civilian_gender': agree(m['civilian_gender_tji'], m['civilian_gender_oag']),
        'civilian_age': agree(m['civilian_age_tji'], m['civilian_age_oag']),
        'civilian_race': agree(m['civilian_race_tji'], m['civilian_race_oag']),
        'civilian_died': agree(m['civilian_died_tji'], m['civilian_died_oag']),
        'deadly_weapon': agree(m['deadly_weapon_tji'].astype(object), m['deadly_weapon_oag'].astype(object)),
        'incident_result_of (raw string)': agree(m['incident_result_of_tji'].map(normalize_call),
                                                 m['incident_result_of_oag']),
        'incident_type_cat (analysis coding)': agree(m['incident_result_of_tji'].map(categorize_incident),
                                                     m['incident_result_of_oag'].map(categorize_incident)),
        'officer_on_duty (single officer)': agree(
            m['officer_on_duty'].astype(object).where(m['num_officers_recorded'] == 1),
            m['officer_on_duty_1'].astype(object).where(m['num_officers_recorded'] == 1)),
    }

    # ---- 3d. Person-level agreement, restricted to persons TJI records as a single row ----
    single = tji[tji['person_key'].map(rows_per_person) == 1].merge(
        persons, on='person_key', suffixes=('_tji', '_oag'))
    # TJI adds officers across a person's reports; in multi-agency shootings each agency often
    # lists every officer who fired, so TJI's sum can double-count. Both rules are checked.
    checks['num_officers, TJI rule = summed (TJI single-row persons)'] = agree(
        single['num_officers_recorded_tji'], single['num_officers_summed'])
    checks['num_officers, distinct officers (TJI single-row persons)'] = agree(
        single['num_officers_recorded_tji'], single['num_officers_recorded_oag'])
    checks['multiple officers (2+) (TJI single-row persons)'] = agree(
        single['num_officers_recorded_tji'] >= 2, single['num_officers_recorded_oag'] >= 2)
    checks['multi_civilian (TJI single-row persons)'] = agree(
        single['num_rows_about_this_incident_tji'] > 1, single['num_rows_about_this_incident_oag'] > 1)

    fields = pd.DataFrame({'field': list(checks),
                           'n': [len(v) for v in checks.values()],
                           'agree': [int(v.sum()) for v in checks.values()]})
    fields['pct_agree'] = (100 * fields['agree'] / fields['n']).round(2)
    print('\nField agreement with TJI:')
    print(fields.to_string(index=False))
    fields.to_csv(STAGING_DIR / 'tx_oag_validation_fields.csv', index=False)

    rep_cols = ['ois_report_no', 'date_incident_tji', 'incident_county_tji', 'civilian_gender_tji',
                'civilian_age_tji', 'civilian_race_tji', 'civilian_race_oag', 'civilian_died_tji',
                'civilian_died_oag', 'deadly_weapon_tji', 'deadly_weapon_oag', 'incident_result_of_tji',
                'incident_result_of_oag', 'person_key_tji']
    per_cols = ['ois_report_no_tji', 'all_report_nos', 'date_incident_tji', 'num_officers_recorded_tji',
                'num_officers_recorded_oag', 'num_officers_summed', 'num_rows_about_this_incident_tji',
                'num_rows_about_this_incident_oag', 'person_key']
    m = m.rename(columns={'person_key_tji': 'person_key_tji'}) if 'person_key_tji' in m else \
        m.rename(columns={'person_key_x': 'person_key_tji'})
    mism = pd.concat(
        [m.loc[~ok, [c for c in rep_cols if c in m]].assign(field=f)
         for f, ok in checks.items() if len(ok) == len(m)]
        + [single.loc[~ok, per_cols].assign(field=f) for f, ok in checks.items() if len(ok) == len(single)])
    mism.to_csv(STAGING_DIR / 'tx_oag_validation_mismatches.csv', index=False)
    print(f'Mismatches written: {len(mism):,}')

    # ---- 3e. Coverage by quarter: OAG persons, those TJI records, and TJI rows ----
    persons['in_tji'] = persons['person_key'].isin(set(tji['person_key'].dropna()))
    ov = persons[persons['date_incident'] <= TJI_END]
    q = lambda d: d.dt.to_period('Q')
    cov = pd.DataFrame({
        'oag_persons': ov.groupby(q(ov['date_incident'])).size(),
        'oag_persons_in_tji': ov[ov['in_tji']].groupby(q(ov['date_incident'])).size(),
        'tji_rows': tji.groupby(q(tji['date_incident'])).size(),
        'tji_rows_unlinked': tji[tji['person_key'].isna()].groupby(q(tji['date_incident'])).size(),
    }).fillna(0).astype(int)
    print('\nCoverage by quarter (overlap through 2024-08-26):')
    print(cov.to_string())
    print(cov.sum().to_string())
    cov.to_csv(STAGING_DIR / 'tx_oag_coverage_by_quarter.csv')
    persons.to_csv(STAGING_DIR / 'tx_oag_persons_all.csv', index=False)

    # ---- 3f. Extension: persons after TJI's last incident through the study end ----
    ext = persons[(persons['date_incident'] > TJI_END) & (persons['date_incident'] <= STUDY_END)].copy()
    ext['weapon_reported_by_media_category'] = '(DETAILS MISSING)'   # TJI media coding not in OAG
    ext['source'] = 'OAG'
    ext.drop(columns=['person_key', 'in_tji']).to_csv(
        STAGING_DIR / 'tx_oag_extension_2024_2025.csv', index=False)
    print(f'\nExtension persons {TJI_END.date()} < date <= {STUDY_END.date()}: {len(ext):,}')
    print(ext.groupby(ext['date_incident'].dt.year).agg(
        persons=('civilian_died', 'size'), fatal=('civilian_died', lambda s: (s == 'DEATH').sum())).to_string())
    print(f'  died conflicts: {ext["died_conflict"].sum()}, weapon conflicts: {ext["weapon_conflict"].sum()}, '
          f'race unparsed: {ext["civilian_race"].isna().sum()}, multi-report persons: '
          f'{(ext["num_reports_filed"] > 1).sum()}')

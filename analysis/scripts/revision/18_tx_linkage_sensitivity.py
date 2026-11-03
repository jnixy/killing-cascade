"""
REVISION (PQ Round 3): Texas record-linkage sensitivity (Reviewer 2, data validation).

R2 asked for sensitivity analyses of two Texas linkage decisions:

  (a) The 22 OAG-reported people from TJI's coverage period left out as likely duplicates of a TJI
      row (00c_build_texas_file.py, Section 5b: an unmatched TJI row in the same county within
      2 days). Here they are added back.
  (b) Officer de-duplication across a person's reports. The OAG reports carry no officer names or
      IDs (checked against the PDFs: officers are listed only by gender, age, race, and duty
      status), so 00b counts each (sex, race, age) officer as many times as it appears on any
      single report. Two officers with identical demographics on the same report are both
      counted; an undercount is possible only when they appear on different reports. Alternative
      rules, from fewest to most officers:
        - current:   distinct (sex, race, age) officers, max multiplicity on any one report
        - rosters:   reports with identical officer rosters are one listing (re-filings, or two
                     agencies listing the same officers); every other listing is a distinct officer
        - summed:    every officer listing on every report is a distinct officer (upper bound;
                     reintroduces the double counting the correction removed)
        - TJI:       TJI's original counts where available (03_robustness.py, spec 20)

Each variant is run through the unchanged 01_prepare_data.py and 02_run_analysis.py (via the
TX_OIS_FILE / TX_OUTPUT_DIR overrides) into a temporary directory, so every estimate comes from the
same code as Table 3. The baseline run must reproduce Table 3 before any variant is reported.

Outputs (outputs/revision/):
  tx_linkage_sensitivity.csv       -- M4 and M5 officer-count and Black coefficients per variant
  tx_linkage_officer_rules.csv     -- how many people each officer-count rule changes
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MERGED_ROOT = SCRIPT_DIR.parent.parent
PROJECT_ROOT = MERGED_ROOT.parent
TX_SCRIPTS = SCRIPT_DIR.parent / 'study1_texas'
STAGING_DIR = PROJECT_ROOT / 'data' / 'processed'  # parsed OAG files ship with the processed data
OUTPUT_DIR = MERGED_ROOT / 'outputs' / 'revision'
TX_FILE = PROJECT_ROOT / 'data' / 'processed' / 'texas_ois_2015_2025.csv'

spec = importlib.util.spec_from_file_location('collapse', TX_SCRIPTS / '00b_validate_and_collapse_oag.py')
collapse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collapse)

tx = pd.read_csv(TX_FILE, low_memory=False, parse_dates=['date_incident'])
persons = pd.read_csv(STAGING_DIR / 'tx_oag_persons_all.csv', parse_dates=['date_incident'])
construction = pd.read_csv(MERGED_ROOT / 'outputs' / 'study1' / 'tables' / 'tx_data_construction.csv').iloc[0]

# =============================================================================
# SECTION 1: OFFICER-COUNT RULES, PERSON BY PERSON
# =============================================================================
parsed = pd.read_csv(STAGING_DIR / 'tx_oag_reports_parsed.csv', low_memory=False, dtype={'incident_street': str},
                     parse_dates=['incident_dt', 'index_incident_date'])
rep = collapse.to_tji_schema(parsed[parsed['Report_Type__c'] == 'Officer Involved'].copy())
rep2person = {r: k for k, nos in zip(persons['person_key'], persons['all_report_nos']) for r in nos.split('/')}
rep['person_key'] = rep['report_no'].map(rep2person)

rules = []
for key, g in rep.dropna(subset=['person_key']).groupby('person_key'):
    rosters = [Counter(str(o) for o in collapse.officers_of(r)) for _, r in g.iterrows()]
    tally = {}
    for c in rosters:
        for o, n in c.items():
            tally[o] = max(tally.get(o, 0), n)
    unique_rosters = []
    for c in rosters:
        if c not in unique_rosters:
            unique_rosters.append(c)
    # Ambiguous: one demographic profile is on two reports whose rosters differ, so it could be
    # one officer listed twice or two officers who look alike
    ambiguous = any(sum(o in c for c in unique_rosters) > 1 for o in tally)
    rules.append({'person_key': key, 'n_reports': len(g), 'n_agencies': g['agency_name'].nunique(),
                  'officers_current': sum(tally.values()),
                  'officers_rosters': sum(sum(c.values()) for c in unique_rosters),
                  'officers_summed': sum(sum(c.values()) for c in rosters),
                  'ambiguous_merge': ambiguous})
rules = pd.DataFrame(rules)
check = persons.merge(rules, on='person_key')
assert (check['num_officers_recorded'] == check['officers_current']).all(), 'current rule not reproduced'
assert (check['num_officers_summed'] == check['officers_summed']).all(), 'summed rule not reproduced'

# Only cases whose count came from the OAG reports are affected by these rules. The extension
# file drops person_key (00b), so those rows are linked by their lead report number.
oag_counted = tx['officer_count_source'] == 'OAG distinct officers'
key_filled = tx['person_key'].isna() & (tx['source'] == 'OAG')
tx.loc[key_filled, 'person_key'] = tx.loc[key_filled, 'ois_report_no'].map(rep2person)
assert tx.loc[oag_counted, 'person_key'].notna().all()
tx = tx.merge(rules[['person_key', 'officers_current', 'officers_rosters', 'officers_summed', 'ambiguous_merge']],
              on='person_key', how='left')
assert (tx.loc[oag_counted, 'officers_current'] == tx.loc[oag_counted, 'num_officers_recorded']).all()


def with_count(col):
    d = tx.copy()
    d['num_officers_recorded'] = np.where(oag_counted, d[col], d['num_officers_recorded']).astype(int)
    return d


# =============================================================================
# SECTION 2: THE 22 LIKELY DUPLICATES (re-derived exactly as in 00c, Section 5b)
# =============================================================================
TJI_END = pd.Timestamp(construction['tji_last_incident'])
tji_rows = tx[tx['source'] == 'TJI']
unmatched_tji = tji_rows[tji_rows['person_key'].isna()].assign(
    county=lambda d: d['incident_county'].str.upper().str.strip())


def has_unmatched_tji_nearby(r):
    return ((unmatched_tji['county'] == str(r['incident_county']).upper().strip())
            & ((unmatched_tji['date_incident'] - r['date_incident']).abs() <= pd.Timedelta('2D'))).any()


cand = persons[(persons['date_incident'] <= TJI_END) & ~persons['person_key'].isin(set(tji_rows['person_key'].dropna()))]
near = cand.apply(has_unmatched_tji_nearby, axis=1)
assert near.sum() == construction['oag_missing_from_tji_likely_dup']
assert (~near).sum() == construction['oag_missing_from_tji_added']
assert set(cand.loc[~near, 'person_key']) == set(tx.loc[tx['source'] == 'OAG (missing from TJI)', 'person_key'])

# Same conversion 00c applies to OAG-only people (as_oag_rows)
likely_dup = cand[near].rename(columns={'num_officers_recorded': 'oag_officers_distinct',
                                        'num_officers_summed': 'oag_officers_summed'}).copy()
likely_dup['num_officers_recorded'] = likely_dup['oag_officers_distinct'].astype(int)
likely_dup['officer_count_source'] = 'OAG distinct officers'
likely_dup['link_method'] = 'OAG source'
likely_dup['n_tji_rows_merged'] = 0
likely_dup['civilian_died'] = likely_dup['civilian_died'].str.upper()
likely_dup['weapon_reported_by_media_category'] = '(DETAILS MISSING)'
likely_dup['source'] = 'OAG (likely duplicate, added back)'
with_dups = pd.concat([tx, likely_dup[[c for c in tx.columns if c in likely_dup.columns]]], ignore_index=True)

# =============================================================================
# SECTION 3: RUN EACH VARIANT THROUGH 01 + 02
# =============================================================================
variants = [
    ('Baseline (as published)', tx),
    ('Add back 22 likely duplicates', with_dups),
    ('Officer count: identical rosters merged only', with_count('officers_rosters')),
    ('Officer count: every listing distinct (summed)', with_count('officers_summed')),
    ('Officer count: TJI original where available', None),
]
models = {'M4': 'Model 4: + Incident Circumstances', 'M5': 'Model 5: + Agency FE'}
out = []
with tempfile.TemporaryDirectory() as tmp:
    for label, data in variants:
        if data is None:
            d = tx.copy()
            d['num_officers_recorded'] = np.where(d['source'] == 'TJI', d['tji_officer_count'],
                                                  d['num_officers_recorded']).astype(int)
            data = d
        vdir = Path(tmp) / f'v{len(out)}'
        (vdir / 'tables').mkdir(parents=True)
        infile = vdir / 'input.csv'
        data = data.drop(columns=['officers_current', 'officers_rosters', 'officers_summed', 'ambiguous_merge'],
                         errors='ignore')
        # Leave the input exactly as 00c wrote it apart from the variant itself
        data.loc[data.index[:len(key_filled)][key_filled.values], 'person_key'] = np.nan
        data.to_csv(infile, index=False)
        env = {**os.environ, 'TX_OIS_FILE': str(infile), 'TX_OUTPUT_DIR': str(vdir)}
        for script in ('01_prepare_data.py', '02_run_analysis.py'):
            subprocess.run([sys.executable, str(TX_SCRIPTS / script)], env=env, check=True,
                           capture_output=True, cwd=MERGED_ROOT)
        reg = pd.read_csv(vdir / 'tables' / 'regression_models.csv')
        n = len(pd.read_csv(vdir / 'texas_regression_data.csv'))
        row = {'variant': label, 'n_input': len(data), 'n_regression': n,
               'mean_officers': pd.read_csv(vdir / 'texas_regression_data.csv')['Officer_Count'].mean()}
        for m, name in models.items():
            r = reg[reg['Model'] == name].iloc[0]
            for var, tag in (('Officer_Count', 'officer'), ('race_black', 'black')):
                row[f'{m}_{tag}_coef'] = r[f'{var}_Coef']
                row[f'{m}_{tag}_ci_lower'] = r[f'{var}_CI_lower']
                row[f'{m}_{tag}_ci_upper'] = r[f'{var}_CI_upper']
                row[f'{m}_{tag}_p'] = r[f'{var}_p']
        out.append(row)
        print(f'{label}: N = {n}, M4 officer {row["M4_officer_coef"]:.2f}, M5 officer {row["M5_officer_coef"]:.2f}, '
              f'M4 Black {row["M4_black_coef"]:.2f}, M5 Black {row["M5_black_coef"]:.2f}')
out = pd.DataFrame(out)

# The baseline must reproduce Table 3 exactly before any variant is trusted
published = pd.read_csv(MERGED_ROOT / 'outputs' / 'study1' / 'tables' / 'regression_models.csv').set_index('Model')
for m, name in models.items():
    for var, tag in (('Officer_Count', 'officer'), ('race_black', 'black')):
        assert np.isclose(out.loc[0, f'{m}_{tag}_coef'], published.loc[name, f'{var}_Coef']), (m, tag)
out.to_csv(OUTPUT_DIR / 'tx_linkage_sensitivity.csv', index=False)

# =============================================================================
# SECTION 4: HOW MANY PEOPLE EACH RULE AFFECTS
# =============================================================================
oag = tx[oag_counted]
summary = pd.DataFrame([{
    'people_total': len(tx),
    'people_count_from_oag': int(oag_counted.sum()),
    'people_multi_report': int((oag['person_key'].map(rules.set_index('person_key')['n_reports']) > 1).sum()),
    'people_any_merge': int((oag['officers_current'] < oag['officers_summed']).sum()),
    'people_ambiguous_merge': int(oag['ambiguous_merge'].sum()),
    'people_rosters_differs': int((oag['officers_rosters'] != oag['officers_current']).sum()),
    'people_summed_differs': int((oag['officers_summed'] != oag['officers_current']).sum()),
    'people_rosters_crosses_1to2': int(((oag['officers_current'] == 1) & (oag['officers_rosters'] >= 2)).sum()),
    'people_summed_crosses_1to2': int(((oag['officers_current'] == 1) & (oag['officers_summed'] >= 2)).sum()),
    'likely_duplicates_added_back': int(near.sum()),
    'likely_duplicates_fatal': int((likely_dup['civilian_died'] == 'DEATH').sum()),
}])
summary.to_csv(OUTPUT_DIR / 'tx_linkage_officer_rules.csv', index=False)
print(summary.T.to_string())

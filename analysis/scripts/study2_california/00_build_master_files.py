"""
Study 2 - California: Rebuild the URSUS master files from data/raw (Phase 1, staging only).

The master files in data/processed/ (Jan 2026) were built by the archived merge/clean
scripts in shared/archive/claude_initial_202501/scripts/. This script reuses those exact
functions so the build is unchanged. Only the input and output folders differ:
  inputs   data/raw/{incident,civilian_officer,ori_agency}/
  outputs  data/staging_2025/   (data/processed/ is replaced only after Phase 2 approval)

Then it checks that the rows for 2016-2024 are identical to the current masters, and
reports the 2025 additions.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
STAGING_DIR = PROJECT_ROOT / 'data' / 'staging_2025'
STAGING_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE = PROJECT_ROOT / 'shared' / 'archive' / 'claude_initial_202501' / 'scripts'


def load(name):
    spec = importlib.util.spec_from_file_location(name, ARCHIVE / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


merge, clean = load('merge_data'), load('clean_data')

# =============================================================================
# SECTION 1: MERGE YEARLY FILES (same calls as merge_data.main)
# =============================================================================
inc = merge.merge_csv_files(list((RAW_DIR / 'incident').glob('*.csv')), 'incident')
civ = merge.merge_csv_files(list((RAW_DIR / 'civilian_officer').glob('*.csv')), 'civilian-officer')
ori = merge.merge_ori_agency_files(list((RAW_DIR / 'ori_agency').glob('*.*')))
inc.to_csv(STAGING_DIR / 'master_incident_data.csv', index=False)
civ.to_csv(STAGING_DIR / 'master_civilian_officer_data.csv', index=False)
ori.to_csv(STAGING_DIR / 'master_ori_agency_mapping.csv', index=False)

# =============================================================================
# SECTION 2: CLEAN AND SPLIT (same calls as clean_data.main)
# =============================================================================
clean.clean_incident_data(STAGING_DIR / 'master_incident_data.csv',
                          STAGING_DIR / 'master_incident_data_clean.csv')
co = clean.clean_civilian_officer_data(STAGING_DIR / 'master_civilian_officer_data.csv',
                                       STAGING_DIR / 'master_civilian_officer_data_clean.csv')
clean.split_civilian_officer_data(co, STAGING_DIR)

# =============================================================================
# SECTION 3: IDENTITY CHECK AGAINST CURRENT MASTERS (2016-2024)
# =============================================================================
print('\n' + '=' * 70)
print('IDENTITY CHECK: staged 2016-2024 rows vs current data/processed masters')
print('=' * 70)
ok = True
for f in ['master_incident_data_clean.csv', 'master_civilian_data_clean.csv',
          'master_officer_data_clean.csv']:
    old = pd.read_csv(PROCESSED_DIR / f, low_memory=False)
    new = pd.read_csv(STAGING_DIR / f, low_memory=False)
    new_old_years = new[new['data_year'] <= 2024]
    # Columns that exist only because 2025 introduced them are all-NA for earlier years.
    extra = [c for c in new_old_years.columns if c not in old.columns]
    extra_nonempty = [c for c in extra if new_old_years[c].notna().any()]
    a = old.reset_index(drop=True)
    b = new_old_years[[c for c in old.columns if c in new_old_years.columns]].reset_index(drop=True)
    same = a.shape == b.shape and a.equals(b)
    ok &= same and not extra_nonempty
    print(f'{f}: old {old.shape}, staged<=2024 {b.shape}, identical={same}; '
          f'new columns {extra} (non-empty pre-2025: {extra_nonempty})')
    print(f'   2025 rows added: {(new["data_year"] == 2025).sum():,}')
old_ori = pd.read_csv(PROCESSED_DIR / 'master_ori_agency_mapping.csv')
print(f'ORI mapping: old {len(old_ori):,} rows, staged {len(ori):,} rows')
print('\nIDENTITY CHECK', 'PASSED' if ok else 'FAILED')
sys.exit(0 if ok else 1)

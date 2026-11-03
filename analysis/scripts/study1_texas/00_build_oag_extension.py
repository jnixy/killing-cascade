"""
Study 1 - Texas: Build the OAG-derived extension to the TJI data (Phase 1, staging only).

TJI's CSV ends 2024-08-26. Incidents after that come from the Texas OAG "Peace Officer
Involved Injuries or Death" reports (CCP Art. 2.139), downloaded from the public portal.

Steps
  1. Pull the full report index from the portal's Visualforce remoting endpoint (the same
     call the page's DataTable makes), so no browser automation is needed.
  2. Download the current version of each report with an incident on or after 2020-01-01
     (cached; a PDF is fetched once).
  3. Parse each PDF's text. The PDFs print only the selected answer to each item, so
     values are read between fixed question labels -- never by searching for option
     keywords, which is what broke the earlier parser (archive/.../extract_pdf_data.py).
     Any field that does not parse is left blank and flagged, never defaulted.
  4. Validate against TJI on report number for the overlap period (written by
     00b_validate_oag_parser.py).
  5. Collapse reports to one row per person shot, in TJI's schema, for incidents
     2024-08-27 through 2025-12-31.

Outputs (Phase 1 -- nothing in outputs/ or data/processed/ is touched):
  data/raw/tx_oag/oag_index_<date>.csv          full portal index as pulled
  data/raw/tx_oag/pdfs/<report>.pdf             cached report PDFs
  data/staging_2025/tx_oag_reports_parsed.csv   one row per report, parsed fields
  data/staging_2025/tx_oag_extension_2024_2025.csv   TJI-schema rows, post-TJI period
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pypdf import PdfReader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
RAW_DIR = PROJECT_ROOT / 'data' / 'raw' / 'tx_oag'
PDF_DIR = RAW_DIR / 'pdfs'
STAGING_DIR = PROJECT_ROOT / 'data' / 'staging_2025'
for p in (PDF_DIR, STAGING_DIR):
    p.mkdir(parents=True, exist_ok=True)

PORTAL = 'https://oag.my.site.com/oisreports'
# Every report in the portal, so officer counts can be re-derived for the whole TJI period
# (TJI rows link to portal report numbers from 2020 on; earlier rows are matched by attributes).
PDF_START = pd.Timestamp('2014-01-01')
TJI_END = pd.Timestamp('2024-08-26')        # last TJI incident date
STUDY_END = pd.Timestamp('2025-12-31')

# =============================================================================
# SECTION 1: PORTAL INDEX
# =============================================================================


def fetch_index():
    """Return every report in the portal index (all versions) as a DataFrame."""
    s = requests.Session()
    page = s.get(f'{PORTAL}/apex/OISReportsPage', timeout=60).text
    cfg = json.loads(re.search(r'RemotingProviderImpl\((\{.*?\})\)\);', page).group(1))
    m = cfg['actions']['OISReports']['ms'][0]
    body = {'action': 'OISReports', 'method': 'getOIS', 'data': [True], 'type': 'rpc', 'tid': 2,
            'ctx': {'csrf': m['csrf'], 'vid': cfg['vf']['vid'], 'ns': '', 'ver': m['ver'],
                    'authorization': m['authorization']}}
    r = s.post(f'{PORTAL}/apexremote', json=body, timeout=180,
               headers={'Referer': f'{PORTAL}/apex/OISReportsPage',
                        'X-User-Agent': 'Visualforce-Remoting'})
    r.raise_for_status()
    idx = pd.DataFrame(r.json()[0]['result'])
    # Portal dates are epoch ms; incident date/time is re-read from the PDF itself below.
    idx['index_incident_date'] = (pd.to_datetime(idx['Date_of_Incident__c'], unit='ms', utc=True)
                                  .dt.tz_convert('America/Chicago').dt.tz_localize(None))
    idx['index_uploaded'] = pd.to_datetime(idx['Date_Uploaded__c'], unit='ms', utc=True).dt.date
    idx['pdf_url'] = (PORTAL + '/VIPForm__VIP_FormWizardPDF?id=' + idx['Id']
                      + '&templateId=' + idx['VIP_Template__c'])
    return idx


# =============================================================================
# SECTION 2: PDF DOWNLOAD
# =============================================================================


def download(row):
    path = PDF_DIR / f"{row['Name']}.pdf"
    if path.exists() and path.stat().st_size > 0:
        return row['Name'], 'cached'
    for attempt in range(3):
        try:
            r = requests.get(row['pdf_url'], timeout=60)
            if r.ok and r.content[:4] == b'%PDF':
                path.write_bytes(r.content)
                return row['Name'], 'downloaded'
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return row['Name'], 'FAILED'


# =============================================================================
# SECTION 3: PDF PARSING
# =============================================================================

PAGE_HEADER = re.compile(r'Peace Officer (?:Involved )?Injur\w* (?:or|and) Death Report :: Page \d+ of \d+')


def pdf_text(path):
    """Whitespace-normalized text of all pages, with running page headers removed."""
    txt = '\n'.join(p.extract_text() or '' for p in PdfReader(path).pages)
    txt = PAGE_HEADER.sub(' ', txt)
    return re.sub(r'\s+', ' ', txt).strip()


def between(txt, start, end):
    """Value printed between two fixed labels; None if either label is absent."""
    m = re.search(start + r'\s*(.*?)\s*' + end, txt, re.IGNORECASE)
    return m.group(1).strip() if m else None


OFFICER_ROW = re.compile(r'(Male|Female)\s+(\d{1,2})?\s*(.*?)\s+(On Duty|Off Duty)', re.IGNORECASE)


def parse_officer_involved(txt):
    d = {
        'agency_name': between(txt, r'Name of Agency/Facility:', r'Street Address:'),
        'version_type_pdf': between(txt, r'Version Type:', r'Name of Agency/Facility:'),
        'civ_gender': between(txt, r"deceased's gender\?:", r'2\. What was'),
        'civ_age': between(txt, r"age at time of incident\?:", r'3\. What was'),
        'civ_race': between(txt, r"race/ethnicity\? \(Mark only one\):", r'Incident Details'),
        'incident_datetime': between(txt, r'4\. Date of Incident:', r'5\. Location of Incident'),
        'incident_street': between(txt, r'5\. Location of Incident Street Address:', r' City:'),
        'incident_city': between(txt, r'5\. Location of Incident Street Address:.*? City:', r'State:'),
        'incident_county': between(txt, r'State: \w* County:', r'Zip:'),
        'incident_zip': between(txt, r'County: .*? Zip:', r'6\. Incident Resulted In'),
        'incident_result': between(txt, r'6\. Incident Resulted In:', r'7\. Injured or Deceased'),
        'civ_weapon': between(txt, r'7\. Injured or Deceased Person:+', r'Peace Officer Information'),
        'multiple_officers_response': between(txt, r'with one or more officers:+', r'13\. Incident occurred'),
        'incident_result_of': between(txt, r'select more than one value\):+', r'$'),
    }
    block = between(txt, r'11\. During the incident, peace officer was:', r'Response/Incident Result Information')
    officers = OFFICER_ROW.findall(block or '')
    d['n_officer_rows'] = len(officers)
    # Cross-check: every officer row ends in a duty status, so the two counts must agree.
    d['n_duty_tokens'] = len(re.findall(r'On Duty|Off Duty', block or '', re.IGNORECASE))
    for i, (g, a, race, duty) in enumerate(officers, start=1):
        d[f'officer_gender_{i}'] = g
        d[f'officer_age_{i}'] = a or None
        d[f'officer_race_{i}'] = race.strip()
        d[f'officer_duty_{i}'] = duty
    return d


def parse_injury_of_officer(txt):
    # Only the date and county are used (to flag officers shot by the civilian).
    return {
        'incident_datetime': re.search(r'Date of Incident:\s*(\S+)', txt).group(1)
        if re.search(r'Date of Incident:\s*(\S+)', txt) else None,
        'incident_county': between(txt, r'County:', r'Zip'),
    }


def parse_one(row):
    path = PDF_DIR / f"{row['Name']}.pdf"
    out = {'report_no': row['Name']}
    try:
        txt = pdf_text(path)
    except Exception as e:      # unreadable PDF is recorded, not dropped
        out['parse_error'] = repr(e)
        return out
    out.update(parse_officer_involved(txt) if row['Report_Type__c'] == 'Officer Involved'
               else parse_injury_of_officer(txt))
    return out


# =============================================================================
# SECTION 4: MAIN
# =============================================================================

if __name__ == '__main__':
    print('Fetching portal index...')
    idx = fetch_index()
    idx.to_csv(RAW_DIR / f'oag_index_{date.today().isoformat()}.csv', index=False)
    print(f'  {len(idx):,} index rows; {idx["Current_Version__c"].sum():,} current versions')

    cur = idx[idx['Current_Version__c']
              & idx['Report_Type__c'].isin(['Officer Involved', 'Injury of Officer'])
              & (idx['index_incident_date'] >= PDF_START)].copy()
    print(f'Downloading {len(cur):,} current-version PDFs (incidents >= {PDF_START.date()})...')
    with ThreadPoolExecutor(max_workers=5) as ex:
        status = {}
        for i, (k, v) in enumerate(ex.map(download, cur.to_dict('records')), start=1):
            status[k] = v
            if i % 100 == 0:
                print(f'  {i:,}/{len(cur):,}', flush=True)
    print('  ', pd.Series(status).value_counts().to_dict())

    print('Parsing PDFs...')
    ok = cur[cur['Name'].map(status) != 'FAILED']
    parsed = pd.DataFrame([parse_one(r) for r in ok.to_dict('records')])
    meta = cur[['Name', 'Report_Type__c', 'Version_Type__c', 'Version_Number__c',
                'Name_of_Agency_Facility__c', 'index_incident_date', 'index_uploaded']]
    parsed = meta.rename(columns={'Name': 'report_no'}).merge(parsed, on='report_no', how='left')
    parsed['incident_dt'] = pd.to_datetime(parsed['incident_datetime'], format='%m/%d/%Y %I:%M %p',
                                           errors='coerce')
    parsed['incident_dt'] = parsed['incident_dt'].fillna(
        pd.to_datetime(parsed['incident_datetime'], format='%m/%d/%Y', errors='coerce'))
    parsed.to_csv(STAGING_DIR / 'tx_oag_reports_parsed.csv', index=False)
    print(f'  Saved {len(parsed):,} parsed reports')

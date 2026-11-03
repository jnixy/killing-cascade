"""
Study 2 - California: Robustness checks.

Rewritten 2026-10-01 for PQ Round 2 on the shared estimator (lpm_utils.fit_lpm): every
specification uses incident-year FE and agency-clustered SEs unless the specification itself
varies the estimator. Baseline = M6 (demographics, armed/assault, incident circumstances,
agency fixed effects, five-category wound location); revised 2026-10-02 when agency FE
entered the main model sequence (Table 4 M5-M7). Output schema and specification labels are unchanged where the
manuscript filters on them.

Added for Round 2: thorax-level wound grouping; officers who used force as the exposure;
officer x CIBRS-era interaction (replaces the Round 1 "consistent across eras" claim).

Output: outputs/study2/tables/robustness_results.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from lpm_utils import fit_lpm  # noqa: E402

MERGED_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = MERGED_ROOT / 'outputs' / 'study2'
TABLES_DIR = OUTPUT_DIR / 'tables'

DEMOG = ['race_black', 'race_hispanic', 'age_10yr', 'female']
CIRC = ['civ_fired_at_officer', 'civ_threat_firearm', 'officer_injured', 'civ_vehicle_against_officer', 'weapon_firearm',
        'weapon_replica', 'weapon_knife', 'weapon_other', 'contact_stop', 'contact_planned',
        'contact_other', 'multi_civilian']
WOUND = ['wound_head_neck', 'wound_chest', 'wound_abdomen', 'wound_back']
BASE = DEMOG + ['armed_assault'] + CIRC + WOUND


def prepare(df):
    """Recreate the 02_run_analysis.py dummies on any subset of the analysis-ready data."""
    d = df.copy()
    d['race_black'] = (d['race_std'] == 'Black').astype(int)
    d['race_hispanic'] = (d['race_std'] == 'Hispanic').astype(int)
    d['female'] = (d['gender_std'] == 'Female').astype(int)
    d['age_10yr'] = d['age_numeric'] / 10
    d['armed_assault'] = d['resistance_std'].isin(['Armed/Deadly Force', 'Physical Assault']).astype(int)
    for col, cat in [('wound_head_neck', 'Head/Neck'), ('wound_chest', 'Chest'),
                     ('wound_abdomen', 'Abdomen'), ('wound_back', 'Back')]:
        d[col] = (d['wound_location_std'] == cat).astype(int)
    for col, cat in [('weapon_firearm', 'Firearm'), ('weapon_replica', 'Replica firearm'),
                     ('weapon_knife', 'Knife/blade'), ('weapon_other', 'Other weapon')]:
        d[col] = (d['weapon_type'] == cat).astype(int)
    for col, cat in [('contact_stop', 'Traffic/pedestrian stop'),
                     ('contact_planned', 'Planned/warrant'), ('contact_other', 'Other/Unknown')]:
        d[col] = (d['contact_reason_cat'] == cat).astype(int)
    d['multi_civilian'] = (d['num_involved_civilians'].fillna(1) > 1).astype(int)
    for col in ['civ_fired_at_officer', 'civ_threat_firearm', 'officer_injured', 'civ_vehicle_against_officer']:
        d[col] = d[col].fillna(0).astype(int)
    d['year'], d['agency'] = d['data_year'], d['ori']
    return d


full = pd.read_csv(OUTPUT_DIR / 'california_analysis_ready.csv')
base_mask = (full['num_involved_officers'].notna() & full['age_numeric'].notna()
             & full['gender_std'].isin(['Male', 'Female']))
reg_df = pd.read_csv(OUTPUT_DIR / 'california_regression_data.csv')
print(f'Regression sample: {len(reg_df):,}')

results = []


def run(data, label, officer='num_involved_officers', covs=None, **kw):
    covs = BASE if covs is None else covs
    kw.setdefault('agency_fe', True)   # baseline M6 includes agency FE
    covs = [c for c in covs if c != officer]
    row = {'specification': label}
    try:
        m = fit_lpm(data, 'fatal', [officer] + covs, **kw)
    except Exception as e:  # noqa: BLE001 - record and continue
        row.update({'n': len(data), 'note': str(e)[:60]})
        results.append(row)
        return row
    row.update({'n': int(m.nobs), 'R2': m.rsquared})
    for var, key in [('race_black', 'Black'), ('race_hispanic', 'Hispanic'), (officer, 'Officer')]:
        if var in m.params.index:
            ci = m.conf_int().loc[var] * 100
            row.update({f'{key}_Coef': m.params[var] * 100, f'{key}_CI_lower': ci[0],
                        f'{key}_CI_upper': ci[1], f'{key}_p': m.pvalues[var]})
    results.append(row)
    print(f"{label:<55} n={row['n']:<5} officer={row.get('Officer_Coef', np.nan):6.2f} "
          f"(p={row.get('Officer_p', np.nan):.3f})  Black={row.get('Black_Coef', np.nan):6.2f}")
    return row


d = reg_df.copy()
d['log_officers'] = np.log(d['num_involved_officers'])
d['officers_capped5'] = d['num_involved_officers'].clip(upper=5)
d['multi_2plus'] = (d['num_involved_officers'] >= 2).astype(int)
d['multi_3plus'] = (d['num_involved_officers'] >= 3).astype(int)

run(d, '1. Baseline (M6: full covariates + agency FE + wound location)')
run(d, '2. Log(officer count)', officer='log_officers')
run(d, '3. Officers capped at 5', officer='officers_capped5')
run(d, '4. Binary (1 vs 2+ officers)', officer='multi_2plus')
run(d, '5. Binary (1-2 vs 3+ officers)', officer='multi_3plus')
run(d, '6. Unadjusted (officer count only)', covs=[])
run(d, '7. Demographics + officer count only', covs=DEMOG)
run(d, '8. Without wound location controls (M5)', covs=DEMOG + ['armed_assault'] + CIRC)
run(d[d['female'] == 0], '9. Males only', covs=[c for c in BASE if c != 'female'])
run(d[d['armed_assault'] == 1], '10. Armed civilians only', covs=[c for c in BASE if c != 'armed_assault'])
run(d[d['age_numeric'] >= 18], '11. Adults only (18+)')
run(d[d['num_involved_officers'] <= 10], '12. Exclude officer count > 10')
run(d[d['num_involved_officers'] >= 2], '13. Multi-officer incidents only')

all_races = prepare(full[base_mask & full['wound_location_std'].isin(
    ['Head/Neck', 'Chest', 'Abdomen', 'Back', 'Extremities'])])
run(all_races, '14. All races (no race controls)', covs=[c for c in BASE if not c.startswith('race_')])
unk = prepare(full[base_mask & full['race_std'].isin(['White', 'Black', 'Hispanic'])])
unk['wound_unknown'] = (unk['wound_location_std'] == 'Unknown').astype(int)
run(unk, '15. Include unknown wound location', covs=BASE + ['wound_unknown'])

run(d[d['data_year'] < 2020], '16. Pre-2020 only')
run(d[d['data_year'] >= 2020], '17. 2020 onwards')
run(d[d['data_year'] <= 2022], '18. Pre-CIBRS (2016-2022)')
run(d[d['data_year'] >= 2023], '19. CIBRS only (2023-2025)')

run(d[d['wound_location_std'].isin(['Head/Neck', 'Chest'])], '20. High-lethality wounds (Head/Chest)',
    covs=[c for c in BASE if c not in ('wound_abdomen', 'wound_back', 'wound_head_neck')])
run(d[d['wound_location_std'].isin(['Abdomen', 'Back', 'Extremities'])],
    '21. Low-lethality wounds (Abdomen/Back/Extremities)',
    covs=[c for c in BASE if c not in ('wound_head_neck', 'wound_chest')])

run(d, '22. HC1 SEs, no fixed effects (Round 1 estimator)', year_fe=False, agency_fe=False, cov='HC1')
# HC3 is undefined with agency FE: single-case agencies have leverage 1, so (1 - h)^2 = 0
run(d, '23. HC3 robust SEs (year FE only)', cov='HC3', agency_fe=False)
run(d, '24. + High-risk regions wounded', covs=BASE + ['wound_count_high'])
run(d, '25. + Wound severity score', covs=BASE + ['wound_severity_score'])
run(d, '26. Without agency fixed effects (year FE only)', agency_fe=False)
run(d[d['armed_assault'] == 0], '27. Unarmed civilians only', covs=[c for c in BASE if c != 'armed_assault'])

# Thorax-level wound grouping (upper torso front + rear = thorax)
t = d.copy()
for col, cat in [('wt_head', 'Head/Neck'), ('wt_thorax', 'Thorax'),
                 ('wt_abdomen_lower_back', 'Abdomen/Lower back'), ('wt_pelvis', 'Pelvis/Buttocks')]:
    t[col] = (t['wound_location_thorax'] == cat).astype(int)
run(t, '28. Thorax-level wound grouping',
    covs=[c for c in BASE if c not in WOUND] + ['wt_head', 'wt_thorax', 'wt_abdomen_lower_back', 'wt_pelvis'])
run(d, '29. Measure: officers who used force', officer='officers_used_force')
run(d, '30. + Regions wounded (M7)', covs=BASE + ['wound_count'])

# Officer x CIBRS-era interaction (Round 1 asserted era consistency without a test)
e = d.copy()
e['cibrs'] = (e['data_year'] >= 2023).astype(int)
e['oc_x_cibrs'] = e['num_involved_officers'] * e['cibrs']
m = fit_lpm(e, 'fatal', ['num_involved_officers', 'oc_x_cibrs'] + BASE, agency_fe=True)
ci = m.conf_int().loc['oc_x_cibrs'] * 100
pd.DataFrame([{'test': 'officer x CIBRS era (2023-25)', 'interaction_pp': m.params['oc_x_cibrs'] * 100,
               'ci_lower': ci[0], 'ci_upper': ci[1], 'p': m.pvalues['oc_x_cibrs'],
               'n_cibrs': int(e['cibrs'].sum()), 'n': int(m.nobs)}]).to_csv(
    TABLES_DIR / 'era_interaction_test.csv', index=False)
print(f"Officer x CIBRS: {m.params['oc_x_cibrs']*100:+.2f} pp [{ci[0]:.2f}, {ci[1]:.2f}], "
      f"p = {m.pvalues['oc_x_cibrs']:.3f}")

robust_df = pd.DataFrame(results)
robust_df.to_csv(TABLES_DIR / 'robustness_results.csv', index=False)
v = robust_df.dropna(subset=['Officer_Coef'])
print(f"\n{len(robust_df)} specifications; officer > 0 in {(v['Officer_Coef'] > 0).sum()}, "
      f"p < .05 in {(v['Officer_p'] < 0.05).sum()}")

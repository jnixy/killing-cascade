"""
REVISION (PQ Round 2): Race and wounds recorded only in rear body regions (California; exploratory).

After the 2026-10-01 wound-coding fix, Black civilians shot by CA police were more likely than
White civilians to have the back as their primary wound region. This script tests that
difference directly.

Orientation measure. URSUS records the body regions struck as front or rear (torso, below
waist, legs); head, neck, and arms carry no orientation. It does not record entry versus exit
or the civilian's posture, so a rear-only pattern does not by itself establish that a bullet
entered from behind (PQ round 3, Reviewer 2).
  - rear_only  = >= 1 rear region and no front region
  - front_only = >= 1 front region and no rear region
Cases with both, or with only unoriented regions (head/neck/arms), are excluded from the
primary contrast.

Specifications (LPM, year FE, agency-clustered SEs; outcome = rear_only):
  1. Race only
  2. + demographics, officer count (more officers -> more chance one is behind the civilian)
  3. + armed/assault, incident circumstances, and flight mentioned in resistance text
Supplements: single-region restriction (wound_count == 1); outcome = any rear region recorded
(all cases); fatality by orientation and race.

Interpretation: being struck from behind is one reading of rear-only wounds, not an established
fact, and it is not the same as "shot while fleeing." Orientation can change in the fraction of a
second between the decision to fire and impact.

Outputs:
  outputs/revision/rear_wound_race_models.csv
  outputs/revision/rear_wound_descriptives.csv
"""

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from lpm_utils import fit_lpm, coef_row  # noqa: E402

MERGED_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = MERGED_ROOT / 'outputs' / 'revision'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# All CA civilians with complete covariates (the regression sample plus the 35 cases whose
# only recorded region is not_applicable are irrelevant here: they have no orientation).
df = pd.read_csv(MERGED_ROOT / 'outputs' / 'study2' / 'california_regression_data.csv')

# Split before normalizing: ', ' separators would otherwise leave a leading '_' on tokens
tokens = df['received_force_location'].astype(str).str.split(',').apply(
    lambda t: [x.strip().upper().replace(' ', '_') for x in t])
df['n_rear'] = tokens.apply(lambda t: sum(x.startswith('REAR') for x in t))
df['n_front'] = tokens.apply(lambda t: sum(x.startswith('FRONT') for x in t))
df['orientation'] = 'Unoriented (head/neck/arms only)'
df.loc[(df['n_rear'] > 0) & (df['n_front'] == 0), 'orientation'] = 'Rear only'
df.loc[(df['n_front'] > 0) & (df['n_rear'] == 0), 'orientation'] = 'Front only'
df.loc[(df['n_front'] > 0) & (df['n_rear'] > 0), 'orientation'] = 'Front and rear'
df['rear_only'] = (df['orientation'] == 'Rear only').astype(int)
df['any_rear'] = (df['n_rear'] > 0).astype(int)

# Flight: mentioned anywhere in the free-text resistance field. resistance_std assigns one
# category with armed/assault taking priority, so it understates flight.
df['flight_mentioned'] = (df['civilian_resistance_type'].astype(str).str.upper()
                          .str.contains('FLEE|FLIGHT|ESCAPE|EVAD|RUN|PURSUIT', regex=True)).astype(int)

circ = ['civ_fired_at_officer', 'civ_threat_firearm', 'officer_injured', 'civ_vehicle_against_officer',
        'weapon_firearm', 'weapon_replica', 'weapon_knife', 'weapon_other',
        'contact_stop', 'contact_planned', 'contact_other', 'multi_civilian']
specs = [
    ('S1: Race only', ['race_black', 'race_hispanic']),
    ('S2: + Demographics, officer count', ['race_black', 'race_hispanic', 'age_10yr', 'female',
                                          'num_involved_officers']),
    ('S3: + Armed, circumstances, flight', ['race_black', 'race_hispanic', 'age_10yr', 'female',
                                           'num_involved_officers', 'armed_assault',
                                           'flight_mentioned'] + circ),
]

oriented = df[df['orientation'].isin(['Rear only', 'Front only'])]
single = oriented[oriented['wound_count'] == 1]
samples = [('Rear-only vs front-only', oriented, 'rear_only'),
           ('Single region only', single, 'rear_only'),
           ('Any rear region (all cases)', df, 'any_rear')]

rows = []
for sample_label, data, outcome in samples:
    for spec_label, xvars in specs:
        res = fit_lpm(data, outcome, xvars)
        for var in ['race_black', 'race_hispanic', 'num_involved_officers', 'flight_mentioned']:
            if var in xvars:
                r = coef_row(res, var)
                rows.append({'sample': sample_label, 'outcome': outcome, 'spec': spec_label,
                             'variable': var, 'N': int(res.nobs),
                             'outcome_mean_white': data.loc[data['race_std'] == 'White', outcome].mean() * 100,
                             **r})

models = pd.DataFrame(rows)
models.to_csv(OUTPUT_DIR / 'rear_wound_race_models.csv', index=False)

# Descriptives: orientation shares by race, and fatality by orientation x race
desc = []
for race in ['Black', 'Hispanic', 'White']:
    r = df[df['race_std'] == race]
    ro = oriented[oriented['race_std'] == race]
    sr = single[single['race_std'] == race]
    desc.append({'race': race, 'n_all': len(r), 'pct_any_rear': r['any_rear'].mean() * 100,
                 'n_oriented': len(ro), 'pct_rear_only_oriented': ro['rear_only'].mean() * 100,
                 'n_single': len(sr), 'pct_rear_only_single': sr['rear_only'].mean() * 100,
                 'pct_flight_mentioned': r['flight_mentioned'].mean() * 100,
                 'cfr_rear_only': r.loc[r['orientation'] == 'Rear only', 'fatal'].mean() * 100,
                 'cfr_front_only': r.loc[r['orientation'] == 'Front only', 'fatal'].mean() * 100})
desc_df = pd.DataFrame(desc)
counts = df['orientation'].value_counts().rename_axis('orientation').reset_index(name='n')
desc_df.to_csv(OUTPUT_DIR / 'rear_wound_descriptives.csv', index=False)
counts.to_csv(OUTPUT_DIR / 'rear_wound_orientation_counts.csv', index=False)

pd.set_option('display.width', 180)
print(counts.to_string(index=False))
print(desc_df.round(1).to_string(index=False))
print(models[models['variable'] == 'race_black'][['sample', 'spec', 'N', 'outcome_mean_white',
                                                   'coef', 'ci_lower', 'ci_upper', 'p']]
      .round(3).to_string(index=False))
print(models[models['variable'].isin(['num_involved_officers', 'flight_mentioned'])
             & models['spec'].str.startswith('S3')].round(3).to_string(index=False))

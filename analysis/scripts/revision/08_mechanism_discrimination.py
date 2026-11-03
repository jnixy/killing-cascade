"""
REVISION: Exploratory moderator analyses.

Revised 2026-10-01 for PQ Round 2. Round 1 framed these as "mechanism discrimination tests."
Reviewer 2 argued, correctly, that the proxies (recorded weapon status, agency) cannot
discriminate among contagious fire, independent threat assessment, coordination failure, and
incident-severity confounding, and that administrative data lack firing sequence. They are now
reported as exploratory moderator analyses only.

Specification: primary (year FE, agency-clustered SEs) on the regression samples; covariates
are demographics only, so the officer-count association is a total association (Round 1's CA
version also adjusted for post-exposure wound location).
  1. Officer count x weapon indicator (TX: deadly weapon; CA: weapon present, with the broader
     armed/assault indicator as a sensitivity check), plus clearly labeled armed-only and
     unarmed-only subgroup slopes.
  2. Agency heterogeneity in the officer-count slope (agencies with >= 20, and >= 10, cases).
     Agency-specific slopes cannot be estimated with agency-clustered SEs, so this joint test
     uses HC1 SEs.

Output: outputs/revision/mechanism_tests.csv (file name kept for downstream references)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from lpm_utils import fit_lpm  # noqa: E402

MERGED_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = MERGED_ROOT / 'outputs' / 'revision'
DEMOG = ['age_10yr', 'female', 'race_black', 'race_hispanic']

rows = []


def weapon_moderation(df, oc, weapon, state, indicator_label):
    d = df.copy()
    d['_oc_x_w'] = d[oc] * d[weapon]
    res = fit_lpm(d, 'fatal', [oc, weapon, '_oc_x_w'] + DEMOG)
    ci = res.conf_int().loc['_oc_x_w'] * 100
    out = {'state': state, 'test': 'officer x weapon interaction', 'indicator': indicator_label,
           'interaction_pp': res.params['_oc_x_w'] * 100, 'interaction_ci_lower': ci[0],
           'interaction_ci_upper': ci[1], 'interaction_p': res.pvalues['_oc_x_w'],
           'n': int(res.nobs), 'pct_armed': d[weapon].mean() * 100}
    for label, sub in [('armed_only', d[d[weapon] == 1]), ('unarmed_only', d[d[weapon] == 0])]:
        r = fit_lpm(sub, 'fatal', [oc] + DEMOG)
        c = r.conf_int().loc[oc] * 100
        out.update({f'{label}_slope_pp': r.params[oc] * 100, f'{label}_ci_lower': c[0],
                    f'{label}_ci_upper': c[1], f'{label}_p': r.pvalues[oc], f'{label}_n': int(r.nobs)})
    print(f"{state} [{indicator_label}]: interaction {out['interaction_pp']:+.2f} pp "
          f"(p = {out['interaction_p']:.3f}); armed-only {out['armed_only_slope_pp']:.2f} "
          f"(n={out['armed_only_n']}), unarmed-only {out['unarmed_only_slope_pp']:.2f} "
          f"(n={out['unarmed_only_n']})")
    return out


def agency_heterogeneity(df, oc, state, min_n):
    counts = df['agency'].value_counts()
    keep = counts[counts >= min_n].index
    d = df[df['agency'].isin(keep)].copy()
    ref = counts[keep].idxmax()
    inter = []
    for i, a in enumerate(keep):
        if a == ref:
            continue
        d[f'_a{i}'] = (d['agency'] == a).astype(float)
        d[f'_a{i}_x'] = d[f'_a{i}'] * d[oc]
        inter.append(f'_a{i}_x')
    dummies = [c[:-2] for c in inter]
    res = fit_lpm(d, 'fatal', [oc, 'armed_ind'] + DEMOG + dummies + inter, cov='HC1')
    w = res.wald_test(', '.join(f'{c} = 0' for c in inter), use_f=True, scalar=True)
    slopes = [fit_lpm(d[d['agency'] == a], 'fatal', [oc], year_fe=False, cov='HC1').params[oc] * 100
              for a in keep if (d['agency'] == a).sum() >= 10]
    out = {'state': state, 'test': f'officer x agency joint F (agencies with {min_n}+ cases)',
           'F_stat': float(w.statistic), 'df_num': int(w.df_num), 'df_den': int(w.df_denom),
           'p_value': float(w.pvalue), 'agency_slope_sd_pp': float(np.std(slopes)),
           'n_agencies_retained': len(keep), 'n': int(res.nobs)}
    print(f"{state} agency heterogeneity (min {min_n}): F({out['df_num']}, {out['df_den']}) = "
          f"{out['F_stat']:.2f}, p = {out['p_value']:.4f}, {len(keep)} agencies, n = {out['n']}")
    return out


tx = pd.read_csv(MERGED_ROOT / 'outputs' / 'study1' / 'texas_regression_data.csv')
ca = pd.read_csv(MERGED_ROOT / 'outputs' / 'study2' / 'california_regression_data.csv')
ca['weapon_present'] = (ca['resistance_std'] == 'Armed/Deadly Force').astype(int)

rows.append(weapon_moderation(tx, 'Officer_Count', 'armed', 'Texas', 'deadly weapon'))
rows.append(weapon_moderation(ca, 'num_involved_officers', 'weapon_present', 'California',
                              'weapon present'))
rows.append(weapon_moderation(ca, 'num_involved_officers', 'armed_assault', 'California',
                              'armed or physical assault (sensitivity)'))

tx['armed_ind'] = tx['armed']
ca['armed_ind'] = ca['weapon_present']
for min_n in (20, 10):
    rows.append(agency_heterogeneity(tx, 'Officer_Count', 'Texas', min_n))
    rows.append(agency_heterogeneity(ca, 'num_involved_officers', 'California', min_n))

pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'mechanism_tests.csv', index=False)
print('Saved: mechanism_tests.csv')

"""
Shared estimation helpers for the PQ Round 2 revision.

Primary specification (adopted 2026-10-01 in response to PQ Reviewer 2): linear probability
model with incident-year fixed effects and standard errors clustered by agency. Civilians shot
in the same incident share an agency, so agency clustering also absorbs within-incident
dependence. Agency fixed effects are available as an option for the sensitivity column.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fe_dummies(df, col, prefix):
    """Drop-first dummies for a fixed-effect column, as float."""
    return pd.get_dummies(df[col].astype(str), prefix=prefix, drop_first=True, dtype=float)


def fit_lpm(df, y, xvars, year_col='year', cluster_col='agency', year_fe=True,
            agency_fe=False, cov='cluster'):
    """Fit an LPM. Returns the statsmodels result; coefficients of xvars are in probability units.

    cov: 'cluster' (by cluster_col), 'HC1', or 'HC3'.
    """
    d = df.dropna(subset=[y] + list(xvars) + [c for c in (year_col, cluster_col) if c]).copy()
    parts = [d[list(xvars)].astype(float)]
    if year_fe:
        parts.append(fe_dummies(d, year_col, 'yr'))
    if agency_fe:
        parts.append(fe_dummies(d, cluster_col, 'ag'))
    X = sm.add_constant(pd.concat(parts, axis=1), has_constant='add')
    model = sm.OLS(d[y].astype(float), X)
    if cov == 'cluster':
        groups = pd.factorize(d[cluster_col])[0]
        res = model.fit(cov_type='cluster', cov_kwds={'groups': groups})
        res.n_clusters = len(np.unique(groups))
    else:
        res = model.fit(cov_type=cov)
        res.n_clusters = np.nan
    res.sample = d
    return res


def coef_row(res, var, scale=100):
    """Coefficient, 95% CI, and p for one variable (scaled to percentage points)."""
    ci = res.conf_int().loc[var]
    return {'coef': res.params[var] * scale, 'ci_lower': ci[0] * scale,
            'ci_upper': ci[1] * scale, 'p': res.pvalues[var]}


def cluster_summary(df, cluster_col='agency'):
    """Number of clusters and cluster-size distribution, for reporting."""
    sizes = df.groupby(cluster_col).size()
    return {'n_clusters': int(sizes.size), 'size_min': int(sizes.min()),
            'size_median': float(sizes.median()), 'size_max': int(sizes.max()),
            'n_singletons': int((sizes == 1).sum()),
            'top5_share': float(sizes.sort_values(ascending=False).head(5).sum() / sizes.sum())}

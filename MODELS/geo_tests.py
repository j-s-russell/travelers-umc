"""Helpers for 17_external_sources: ZIP-level residual tests, density-variant joins, formula refits on alternative inputs."""
import numpy as np
import pandas as pd

import external as E
import formula as FM


def add_density(df, T=None):
    """Join every D_* density variant (and LAT/LON/COUNTY_FIPS/WATER_SHARE) on ZIP_CD. Missing ZIP -> NaN (median-imputed by Formula)."""
    T = E.density_table() if T is None else T
    z = df.ZIP_CD.astype(str).where(df.ZIP_CD.astype(str).str.fullmatch(r'\d{5}'))
    J = T.reindex(z.to_numpy())
    J.index = df.index
    return df.assign(**{c: J[c] for c in J})


def zip_effect(y, p, groups, eras=None):
    """Moment test for a random intercept per group on the logit scale, given fitted probabilities p.
    S_g = sum of residuals in group g, V_g = sum p(1-p). Under the null E[S_g^2] = V_g.
    Returns {'stat': sum(S^2 - V) / sqrt(2 sum V^2), 'tau2': sum(S^2 - V) / sum(V^2)} (tau2 ~ variance of the group log-odds effect)."""
    r, w = y - p, p * (1 - p)
    out = {}
    keys = {'all': np.ones(len(y), bool)} if eras is None else {'all': np.ones(len(y), bool)} | {e: eras == e for e in np.unique(eras)}
    for k, m in keys.items():
        d = pd.DataFrame({'g': groups[m], 'r': r[m], 'w': w[m]}).groupby('g').sum()
        num = (d.r ** 2 - d.w).sum()
        out[k] = {'stat': num / np.sqrt(2 * (d.w ** 2).sum()), 'tau2': num / (d.w ** 2).sum()}
    return out


def fit_predict(spec, L, y):
    m = FM.Formula(spec, 0.1, '2era').fit(L, y)
    return m, m.predict(L)


def deviance(spec, L, y):
    m, p = fit_predict(spec, L, y)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -2 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)), m


def swap(spec, old, new):
    return [dict(x, col=new) if x['col'] == old else x for x in spec]

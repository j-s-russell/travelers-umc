"""Screen area-level and timing candidates for the 2017-18 regional effect left by SPEC8 (17_external_sources §8).
Each candidate is a column active only in 2017-18; score test against the fitted base formula (all candidates at once, vectorized).
Pure-chance threshold: 95th percentile of the max chi2 over outcome sets simulated from the base formula.
Run from models/: area_screen.py N_SIM OUT"""
import sys, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

import external as E
import formula as FM
import formula_steps as FS

FIPS = {'01', '09', '12', '13', '27', '34', '36', '55'}


def area_candidates(L):
    V = E.area_variables()
    V['acs18_' + 'density'] = E.acs_pop(2018).POP / (E.gazetteer(2018).ALAND / 1e6)
    a18 = pd.read_parquet(E.EXT / 'acs2018_zcta_features.parquet')
    for c in a18:
        V['acs18_' + c[4:].lower()] = a18[c]
    rel = E.zcta_county_2010()
    rel = rel[rel.STATE.isin(FIPS)].sort_values('ZPOPPCT', ascending=False).drop_duplicates('ZCTA').set_index('ZCTA')
    zc = pd.Index(rel.index)
    W = V.reindex(zc)
    pop = E.gazetteer(2010).POP10.reindex(zc).astype(float).fillna(0)
    def wmean(key):
        k = pd.Series(key, index=zc)
        num = W.mul(pop, axis=0).groupby(k).sum(min_count=1)
        den = W.notna().mul(pop, axis=0).groupby(k).sum()
        return (num / den.replace(0, np.nan)).reindex(k).set_axis(zc)
    levels = {'zip': W, 'zip3': wmean(zc.str[:3]), 'county': wmean(rel.GEOID.to_numpy())}
    z = L.ZIP_CD.astype(str).where(L.ZIP_CD.astype(str).str.fullmatch(r'\d{5}')).to_numpy()
    cols = {}
    for lvl, T in levels.items():
        J = T.reindex(z)
        for c in T:
            x = J[c].to_numpy(float)
            cols[f'{c} [{lvl}]'] = x
            if np.nanmin(x) >= 0:
                cols[f'log {c} [{lvl}]'] = np.log1p(x)
    return pd.DataFrame(cols, index=L.index)


def timing_candidates(L):
    dt = pd.to_datetime(L.QUOTE_DT)
    out = {'days since 2017-01-01': (dt - pd.Timestamp('2017-01-01')).dt.days.astype(float), 'year 2018': (dt.dt.year == 2018).astype(float),
           'month of year': dt.dt.month.astype(float), 'day of week': dt.dt.dayofweek.astype(float)}
    for q in pd.period_range('2017Q1', '2018Q4', freq='Q'):
        out[f'quarter {q}'] = (dt.dt.to_period('Q') == q).astype(float)
    for h in (2017, 2018):
        out[f'second half {h}'] = ((dt.dt.year == h) & (dt.dt.month >= 7)).astype(float)
    return pd.DataFrame(out, index=L.index)


def score_chi2(spec, L, y, Xc):
    """Score-test chi2 for adding each column of Xc (already restricted to the late era) to the fitted formula."""
    m = FM.Formula(spec, 0.1, '2era').fit(L, y)
    p = m.predict(L)
    X = np.column_stack([np.ones(len(y)), m.sc.transform(m._design(L))])
    w = p * (1 - p)
    Minv = np.linalg.pinv(X.T @ (X * w[:, None]))
    B = X.T @ (Xc * w[:, None])
    info = (Xc ** 2 * w[:, None]).sum(0) - np.einsum('ij,ij->j', B, Minv @ B)
    U = Xc.T @ (y - p)
    chi = np.where(info > 5, U ** 2 / np.maximum(info, 1e-12), 0.0)
    return chi, U


def build(L):
    late = (L.ERA.astype(str) == '2017-18').to_numpy()
    C = pd.concat([area_candidates(L), timing_candidates(L)], axis=1)
    C = C.loc[:, C.notna().mean() > 0.9]
    Xc = C.fillna(C.median()).to_numpy(float)
    Xc = (Xc - Xc[late].mean(0)) / (Xc[late].std(0) + 1e-12)
    Xc[~late] = 0.0
    return C.columns, Xc


def main():
    n_sim, out = int(sys.argv[1]), sys.argv[2]
    L, y = FS.L, FS.y
    names, Xc = build(L)
    real, U = score_chi2(FS.SPEC8, L, y, Xc)
    m = FM.Formula(FS.SPEC8, 0.1, '2era').fit(L, y)
    p = m.predict(L)
    rng = np.random.default_rng(17)
    null = []
    for i in range(n_sim):
        ys = (rng.random(len(y)) < p).astype(float)
        null.append(score_chi2(FS.SPEC8, L.assign(CONVERT_IND=ys), ys, Xc)[0].max())
    res = {'names': list(names), 'chi2': real, 'U': U, 'null_max': np.array(null)}
    pickle.dump(res, open(out, 'wb'))
    thr = np.quantile(null, 0.95) if null else np.nan
    print(f'{len(names)} candidates; pure-chance threshold (95th pct of max over {n_sim} sims): {thr:.1f}')
    print(pd.Series(real, index=names).sort_values(ascending=False).head(25).round(1).to_string())


if __name__ == '__main__':
    main()

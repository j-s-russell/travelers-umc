"""Target-free imputation variants for the formula (17_external_sources). Missing values look masked after the outcome was
generated (residuals on missing rows ~ 0), so the best input is the expected value of each term given what is observed.
For step/flag terms that means the probability the hidden value crosses the cut, used as a fractional flag.
`add(df, all_rows)` adds the columns below; imputers are fit on features of `all_rows` (labeled + holdout, no target)."""
import numpy as np
import pandas as pd
import lightgbm as lgb

from common import ROOT
import external as E

DRV = ROOT / 'data/processed/drivers.parquet'
VEH = ROOT / 'data/processed/vehicles.parquet'


def density_km2(df):
    """Generator density: 2010 Census population / 2018 Gazetteer land area, people per km^2 (ZIPs were selected at >= 1000)."""
    z = df.ZIP_CD.astype(str).where(df.ZIP_CD.astype(str).str.fullmatch(r'\d{5}')).to_numpy()
    pop, land = E.gazetteer(2010).POP10.astype(float), E.gazetteer(2018).ALAND / 1e6
    return pd.Series((pop.reindex(z) / land.reindex(z)).to_numpy(), index=df.index)


def _edu_majority_prob(pids):
    """P(share of educated drivers > 0.5), unknown drivers ~ Bernoulli(base rate)."""
    d = pd.read_parquet(DRV, columns=['POLICY_ID', 'HIGH_EDU_IND'])
    rate = d.HIGH_EDU_IND.mean()
    g = d.groupby('POLICY_ID').HIGH_EDU_IND.agg(known='count', edu='sum', n='size')
    u = (g.n - g.known).astype(int)
    from scipy.stats import binom
    need = np.floor(g.n / 2).astype(int) + 1 - g.edu          # extra educated drivers needed for a strict majority
    prob = np.where(need <= 0, 1.0, binom.sf(need - 1, u, rate))
    return pd.Series(prob, index=g.index).reindex(pids).to_numpy()


def _veh_min_ge5_prob(pids):
    """P(newest vehicle >= 5 years), unknown vehicle ages ~ empirical P(age >= 5)."""
    v = pd.read_parquet(VEH, columns=['POLICY_ID', 'VEHICLE_AGE'])
    q = (v.VEHICLE_AGE >= 5).mean() / v.VEHICLE_AGE.notna().mean()
    g = v.groupby('POLICY_ID').VEHICLE_AGE.agg(kmin='min', known='count', n='size')
    u = g.n - g.known
    known_ok = g.kmin.ge(5) | g.known.eq(0)
    return pd.Series(np.where(known_ok, q ** u, 0.0), index=g.index).reindex(pids).to_numpy()


def _price_model(all_rows):
    """LightGBM regression of log price on the other policy/driver/vehicle columns (target-free)."""
    feats = [c for c in all_rows if c not in ('POLICY_ID', 'SPLIT', 'CONVERT_IND', 'QUOTE_DT', 'ZIP_CD', 'ZIP3', 'QUOTED_AMT',
                                               'MONTHLY_PREMIUM', 'QUOTED_AMT_MISSING', 'PRICE_PER_CREDIT', 'QUOTE_PER_DRIVER', 'QUOTE_PER_VEH')
             and not c.startswith('V2_')]   # price-derived columns excluded (they leak price)
    X = all_rows[feats].copy()
    for c in X:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype('category')
    ok = all_rows.QUOTED_AMT.notna().to_numpy()
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1).fit(X[ok], np.log(all_rows.QUOTED_AMT[ok]))
    return m, feats


def core(d):
    """Columns adopted into the formula (SPEC8), computed without the target or other rows: generator density, education
    majority and newest-vehicle-age >= 5 as probabilities over missing values, package ordinal with missing -> expected value."""
    d = d.copy()
    d['DEN_KM2'] = density_km2(d)
    d['R_EDU_MAJ_P'] = _edu_majority_prob(d.POLICY_ID)
    d['VEH_MIN_GE5_P'] = _veh_min_ge5_prob(d.POLICY_ID)
    pk = pd.read_parquet(ROOT / 'data/processed/policies.parquet', columns=['COV_PKG_TYPE']).COV_PKG_TYPE.astype(str)
    m = {'Low': -1, 'Medium': 0, 'High': 1}
    d['PKG_ORDINAL_IMP'] = d.COV_PKG_TYPE.astype(str).map(m).fillna(pk.map(m).mean())
    return d


def add(df, all_rows):
    d = df.copy()
    d['DEN_KM2'] = density_km2(d)
    A = all_rows.copy()
    A['DEN_KM2'] = density_km2(A)
    st = A.groupby('STATE_CD', observed=True)
    # ZIP missing -> state mean density; CAT missing -> state mean CAT zone (CAT is drawn per state)
    d['DEN_KM2_IMP'] = d.DEN_KM2.fillna(d.STATE_CD.astype(str).map(st.DEN_KM2.mean().rename(index=str)))
    cz = pd.to_numeric(d.CAT_ZONE, errors='coerce').astype(float)
    d['CAT_ZONE_IMP'] = cz.fillna(d.STATE_CD.astype(str).map(pd.to_numeric(A.CAT_ZONE, errors='coerce').astype(float).groupby(A.STATE_CD.astype(str)).mean()))
    # package: expected ordinal
    po = A.COV_PKG_TYPE.astype(str).map({'Low': -1, 'Medium': 0, 'High': 1})
    d['PKG_ORDINAL_IMP'] = d.COV_PKG_TYPE.astype(str).map({'Low': -1, 'Medium': 0, 'High': 1}).fillna(po.mean())
    # credit >= 650 as a probability when missing
    cs = pd.to_numeric(A.CREDIT_SCORE, errors='coerce').astype(float)
    d['CREDIT_GE650_P'] = np.where(d.CREDIT_SCORE.isna(), (cs >= 650).sum() / cs.notna().sum(), (d.CREDIT_SCORE.astype(float) >= 650).astype(float))
    d['CREDIT_GE650'] = (d.CREDIT_SCORE.astype(float).fillna(cs.median()) >= 650).astype(float)
    # education majority and newest-vehicle-age >= 5 as probabilities
    d['R_EDU_MAJ_P'] = _edu_majority_prob(d.POLICY_ID)
    d['VEH_MIN_GE5_P'] = _veh_min_ge5_prob(d.POLICY_ID)
    # price: model imputation
    m, feats = _price_model(A)
    X = d[feats].copy()
    for c in X:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.Categorical(X[c], categories=A[c].astype('category').cat.categories)
    d['QUOTED_AMT_IMP'] = d.QUOTED_AMT.astype(float).fillna(pd.Series(np.exp(m.predict(X)), index=d.index))
    return d

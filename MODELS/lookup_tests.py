"""Helpers for 18_lookup_sources: is a variable a lookup on some key, a draw from a key-conditional table, or independent?
dependence(x, key): share of x's variation explained by key (eta^2 for numeric x, Goodman-Kruskal tau for categorical x),
compared with a permutation null (key shuffled). Also tables at policy / driver / vehicle level with candidate keys."""
import numpy as np
import pandas as pd

from common import ROOT


def _explained(x, key, categorical):
    k = pd.Series(key).astype(str).to_numpy()
    if categorical:
        x = pd.Series(x).astype(str).to_numpy()
        tab = pd.crosstab(k, x).to_numpy(float)
        n = tab.sum()
        gini = 1 - ((tab.sum(0) / n) ** 2).sum()
        rows = tab.sum(1)
        cond = (rows / n * (1 - ((tab / rows[:, None]) ** 2).sum(1))).sum()
        return (gini - cond) / gini
    x = np.asarray(x, float)
    s = pd.DataFrame({'x': x, 'k': k})
    m = s.groupby('k').x.transform('mean')
    return 1 - ((s.x - m) ** 2).sum() / ((s.x - s.x.mean()) ** 2).sum()


def dependence(x, key, categorical=False, n_perm=20, seed=0):
    ok = pd.notna(x) & pd.notna(key)
    x, key = np.asarray(x)[ok], np.asarray(key)[ok]
    e = _explained(x, key, categorical)
    rng = np.random.default_rng(seed)
    null = np.array([_explained(x, rng.permutation(key), categorical) for _ in range(n_perm)])
    nk = len(pd.unique(key))
    return {'explained': e, 'null': null.mean(), 'excess': e - null.mean(), 'z': (e - null.mean()) / (null.std() + 1e-12), 'levels': nk}


def screen(df, targets, keys, categorical, n_perm=20):
    rows = []
    for tgt in targets:
        for k in keys:
            if k == tgt:
                continue
            r = dependence(df[tgt].to_numpy(), df[k].to_numpy(), categorical=tgt in categorical, n_perm=n_perm)
            rows.append({'variable': tgt, 'key': k} | r)
    return pd.DataFrame(rows)


def shape(s):
    """Marginal fingerprint: distinct values, range, rounding grid, top values."""
    s = pd.Series(s).dropna()
    out = {'n': len(s), 'distinct': s.nunique()}
    if pd.api.types.is_numeric_dtype(s):
        v = s.to_numpy(float)
        grid = next((g for g in (1000, 500, 100, 50, 10, 5, 1, 0.5, 0.1, 0.01) if np.allclose(v / g, np.round(v / g))), None)
        out |= {'min': v.min(), 'p1': np.quantile(v, .01), 'median': np.median(v), 'p99': np.quantile(v, .99), 'max': v.max(),
                'mean': v.mean(), 'sd': v.std(), 'skew': pd.Series(v).skew(), 'grid': grid,
                'share at min': (v == v.min()).mean(), 'share at max': (v == v.max()).mean()}
    out['top'] = s.value_counts(normalize=True).head(5).round(3).to_dict()
    return out


def tables():
    """Policy-, driver- and vehicle-level frames with candidate keys attached (all splits, no target)."""
    import common as C, features as FE
    lab, hold = C.load()
    P = FE.add_era(FE.add_static(pd.concat([lab, hold])))
    P['ZIP3'] = P.ZIP_CD.astype(str).str[:3]
    P['YEAR'] = P.QUOTE_YEAR
    P['CREDIT_BAND'] = pd.cut(P.CREDIT_SCORE.astype(float), [0, 550, 600, 650, 700, 750, 900]).astype(str)
    pk = ['POLICY_ID', 'STATE_CD', 'ZIP_CD', 'ZIP3', 'YEAR', 'ERA', 'NUM_DRIVERS', 'VEH_N_ROWS', 'COV_PKG_TYPE', 'PRIOR_CARRIER_GRP',
          'CAT_ZONE', 'QUOTE_CHANNEL', 'DISCOUNT_IND', 'HOME_POLICY_IND', 'CREDIT_BAND']
    D = pd.read_parquet(ROOT / 'data/processed/drivers.parquet')
    D['DRIVER_POS'] = D.groupby('POLICY_ID').cumcount() + 1
    D['AGE_BAND'] = pd.cut(D.DRIVER_AGE, [0, 18, 20, 22, 25, 30, 35, 45, 55, 65, 75, 200]).astype(str)
    D['ACC_FREE_BAND'] = pd.cut(D.ACCIDENT_FREE_YRS, [-1, 2, 5, 10, 15, 20]).astype(str)
    D = D.merge(P[pk], on='POLICY_ID', how='left')
    V = pd.read_parquet(ROOT / 'data/processed/vehicles.parquet')
    V['VEH_AGE_KEY'] = V.VEHICLE_AGE.astype('Int64').astype(str)
    V = V.merge(P[pk], on='POLICY_ID', how='left')
    return P, D, V


STATE_ABBR = {'Alabama': 'AL', 'Connecticut': 'CT', 'Florida': 'FL', 'Georgia': 'GA', 'Minnesota': 'MN', 'New Jersey': 'NJ',
              'New York': 'NY', 'Wisconsin': 'WI'}


def naic_top10(year):
    """Top-10 group market shares by state, Total Private Passenger Auto, parsed from the NAIC Market Share Report PDF
    (data/external/manual/NAIC_<year>.pdf; 2010 and 2017 layouts). Cached as data/external/naic<year>_ppa_top10_by_state.csv."""
    import re
    f = ROOT / f'data/external/naic{year}_ppa_top10_by_state.csv'
    if f.exists():
        return pd.read_csv(f)
    from pypdf import PdfReader
    pages = [p.extract_text() or '' for p in PdfReader(ROOT / f'data/external/manual/NAIC_{year}.pdf').pages]
    rows, state = [], None
    for t in pages:
        head = t[:400]
        if 'Total Private Passenger Auto' not in head:
            continue
        if year >= 2017 and 'BY STATE BY GROUP' not in head:
            continue
        for line in t.split('\n'):
            line = line.strip()
            if year >= 2017:
                m = re.match(r'^(\d+)\s+(\d+)\s+(.+?)\s+%(\d+\.\d{2})%(\d+\.\d{2})', line)
                if not m:
                    continue
                if m.group(1) == '1':
                    s = re.search(r'\d([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+\d', line)
                    state = s.group(1) if s else None
                rows.append({'state': state, 'rank': int(m.group(1)), 'group': m.group(3).replace('TRA VELERS', 'TRAVELERS').strip(), 'share': float(m.group(5))})
            else:
                m = re.match(r'^(?:([A-Z][a-z]+(?: [A-Z][a-z]+)*) )?(\d+) (\d+) (.+?) [\d,]+ [\d,]+ -?[\d.]+ -?[\d.]+ (\d+\.\d+) (\d+\.\d+)$', line)
                if not m:
                    continue
                if m.group(1):
                    state = m.group(1)
                elif m.group(2) == '1':
                    state = None
                rows.append({'state': state, 'rank': int(m.group(2)), 'group': m.group(4).strip(), 'share': float(m.group(5))})
    N = pd.DataFrame(rows)
    N = N[N.state.isin(STATE_ABBR)].assign(ST=lambda d: d.state.map(STATE_ABBR))
    N.to_csv(f, index=False)
    return N


def carrier_match(real, ours, n_perm=1000, seed=0):
    """Best one-to-one assignment of our carrier labels to real groups (L1 distance of state share profiles), vs a null where each
    carrier's state profile is shuffled across states. real: rows (group, ST, share); ours: carriers x states, % of quotes."""
    from scipy.optimize import linear_sum_assignment
    R = real.pivot_table(index='group', columns='ST', values='share').reindex(columns=ours.columns).fillna(0.0)
    def best(o):
        C = np.array([[np.abs(o.loc[c].values - R.loc[g].values).sum() for g in R.index] for c in o.index])
        r, c = linear_sum_assignment(C)
        return C[r, c].sum(), {o.index[a]: R.index[b] for a, b in zip(r, c)}
    cost, assign = best(ours)
    rng = np.random.default_rng(seed)
    null = np.array([best(pd.DataFrame([rng.permutation(v) for v in ours.values], index=ours.index, columns=ours.columns))[0] for _ in range(n_perm)])
    return {'cost': cost, 'null_mean': null.mean(), 'null_p05': np.quantile(null, 0.05), 'p': (null <= cost).mean(), 'assign': assign, 'R': R}

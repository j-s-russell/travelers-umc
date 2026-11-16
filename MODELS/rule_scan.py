"""Exhaustive rule scan for the generator formula (16_rule_scan).

Idea: given the current formula's fitted probabilities p, the **score test** for adding one candidate column x is
    U = x'(y - p),   V = x'Wx - (X'Wx)' (X'WX)^-1 (X'Wx),   chi2 = U^2 / V      (W = p(1 - p), X = current design + intercept)
which needs no refit, so millions of candidate rules can be ranked quickly. V accounts for overlap with existing terms.

Candidate families (each multiplied by an era scope: all / early (2015-16) / late (2017-18) / 2015 / 2016):
- 'band':   a <= x <= b for every pair of cut points of every numeric column (incl. one-sided steps and a missing flag);
            numeric columns include policy, Census, roll-ups, and driver/vehicle row-level predicate counts and shares
            (e.g. number / share of drivers with ACCIDENT_FREE_YRS >= 5), plus first-driver / first-vehicle values.
- 'subset': every subset of the levels of a categorical column (<= 12 levels; single levels otherwise).
- 'pair':   step x step (or step x level) for every pair of a base set of columns, scopes all / early / late.

A rule is stored as a dict and can be materialized on any frame (labeled or holdout) with `materialize`.
"""
import itertools

import numpy as np
import pandas as pd

import common as C

SCOPES = ['all', 'early', 'late', '2015', '2016']
PAIR_SCOPES = ['all', 'early', 'late']
TOP_PER_BLOCK = 20   # candidates kept per (column, scope); family maxima are exact


def scope_mask(d, s):
    if s == 'all':
        return np.ones(len(d))
    if s == 'early':
        return d.ERA.isin(['2015', '2016']).to_numpy(float)
    if s == 'late':
        return d.ERA.eq('2017-18').to_numpy(float)
    return d.ERA.eq(s).to_numpy(float)


# ---------------------------------------------------------------- candidate columns
DRV_NUM = ['DRIVER_AGE', 'SAFETY_RATING', 'ACCIDENT_FREE_YRS', 'COMMUTE_MILES_DAILY']
DRV_CAT = {'GENDER': None, 'MARITAL_STATUS': None, 'LIVING_STATUS': None, 'HIGH_EDU_IND': None, 'TELEMATICS_OPT_IN': None}
VEH_NUM = ['VEHICLE_AGE', 'ANNUAL_MILEAGE', 'SAFETY_FEATURE_SCORE']
VEH_CAT = {'OWNERSHIP_TYPE': None, 'COLOR': None, 'ANTI_THEFT_IND': None, 'GARAGE_IND': None, 'MAKE': None}
EXCLUDE = {'CONVERT_IND', 'QUOTE_DT', 'SPLIT', 'POLICY_ID', 'ZIP_CD', 'CARRIER_STATE', 'PKG_CAT'}
CAT_COLS = ['STATE_CD', 'PRIOR_CARRIER_GRP', 'COV_PKG_TYPE', 'PRIMARY_PARKING', 'QUOTE_CHANNEL', 'ZIP3', 'ERA']


def _row_thresholds(s, k=12):
    v = s.dropna().to_numpy(float)
    u = np.unique(v)
    return u[1:] if len(u) <= k + 1 else np.unique(np.quantile(v, np.linspace(0.05, 0.95, k)))


def build_columns(d):
    """All candidate columns for the policies in `d` (target-free; identical construction for labeled and holdout).
    Returns (num: {name: float array with NaN}, cat: {name: str array})."""
    num = {c: d[c].astype(float).to_numpy() for c in d.columns
           if c not in EXCLUDE and pd.api.types.is_numeric_dtype(d[c]) and not c.startswith(('ALT_', 'X3_', 'AGEB_'))}
    cat = {c: d[c].astype(object).where(d[c].notna(), 'Missing').astype(str).to_numpy() for c in CAT_COLS if c in d}
    ids = d.POLICY_ID
    nd = d.NUM_DRIVERS.to_numpy(float)
    dr = pd.read_parquet(C.PROC / 'drivers.parquet')
    vh = pd.read_parquet(C.PROC / 'vehicles.parquet')
    nv = d.VEH_N_ROWS.to_numpy(float)
    for tab, nums, cats, tag, denom in ((dr, DRV_NUM, DRV_CAT, 'DRV', nd), (vh, VEH_NUM, VEH_CAT, 'VEH', nv)):
        preds = {}
        for c in nums:
            for th in _row_thresholds(tab[c]):
                preds[f'{c}>={th:g}'] = tab[c] >= th
                preds[f'{c}<{th:g}'] = tab[c] < th
        for c in cats:
            for lv in tab[c].astype(str).unique():
                preds[f'{c}={lv}'] = tab[c].astype(str).eq(lv)
        P = pd.DataFrame(preds).astype(float).assign(POLICY_ID=tab.POLICY_ID.to_numpy()).groupby('POLICY_ID').sum()
        P = P.reindex(ids).to_numpy()
        names = list(preds)
        for j, n in enumerate(names):
            num[f'{tag}_N[{n}]'] = P[:, j]
            num[f'{tag}_SHARE[{n}]'] = P[:, j] / np.where(denom > 0, denom, np.nan)
        first = tab.groupby('POLICY_ID').head(1).set_index('POLICY_ID') if tag == 'DRV' else tab[tab.CAR_NO == 1].set_index('POLICY_ID')
        for c in nums:
            num[f'{tag}_FIRST[{c}]'] = first[c].reindex(ids).astype(float).to_numpy()
        for c in cats:
            cat[f'{tag}_FIRST[{c}]'] = first[c].reindex(ids).astype(object).where(first[c].reindex(ids).notna(), 'Missing').astype(str).to_numpy()
    return num, cat


def cut_grid(x, k=40):
    v = x[~np.isnan(x)]
    u = np.unique(v)
    if len(u) <= 1:
        return np.array([])
    if len(u) <= k + 1:
        return u[1:]                                          # steps between every pair of observed values
    return np.unique(np.quantile(v, np.linspace(0.025, 0.975, k)))


# ---------------------------------------------------------------- score-test machinery
class Context:
    def __init__(self, X, p, y):
        X = np.column_stack([np.ones(len(y)), X])
        self.W = p * (1 - p)
        self.r = y - p
        self.XW = X * self.W[:, None]
        self.Q = np.linalg.pinv(X.T @ self.XW)

    def parts(self, B):
        """U, x'Wx, A = X'Wx for indicator columns B (n x m)."""
        return B.T @ self.r, B.T @ self.W, self.XW.T @ B


MIN_INFO = 5.0   # minimum independent information V (~50 quotes at p(1-p) ~ 0.1): rules that differ from existing terms
                 # by a handful of quotes make the score test unreliable, so they are skipped


def _chi2(U, WS, quad):
    V = WS - quad
    return np.where(V > MIN_INFO, U ** 2 / np.maximum(V, 1e-12), 0.0), U


def scan_bands(ctx, x, masks, name, grid):
    """All bands [a, b] (a, b in grid, a <= b; a = -inf and b = +inf allowed) and a missing flag, in every scope."""
    out = []
    nn = ~np.isnan(x)
    xv = np.where(nn, x, np.nan)
    GE = (xv[:, None] >= grid[None, :]).astype(float)            # n x k   (NaN compares False)
    GT = (xv[:, None] > grid[None, :]).astype(float)
    k = len(grid)
    for s, m in masks.items():
        Lw = np.column_stack([nn * m, GE * m[:, None]])          # lower options: -inf, grid
        Uw = GT * m[:, None]                                     # upper exclusions: grid (b=+inf -> nothing)
        UL, WL, AL = ctx.parts(Lw)
        UU, WU, AU = ctx.parts(Uw)
        QAL, QAU = ctx.Q @ AL, ctx.Q @ AU
        LL = np.einsum('pi,pi->i', AL, QAL)                      # (k+1)
        UUq = np.einsum('pi,pi->i', AU, QAU)                     # (k)
        LU = AL.T @ QAU                                          # (k+1) x k
        # column b = k means b = +inf (no upper exclusion); lower a: index 0 = -inf, a >= 1 = grid[a-1]
        U2 = np.column_stack([UL[:, None] - UU[None, :], UL])
        W2 = np.column_stack([WL[:, None] - WU[None, :], WL])
        q2 = np.column_stack([LL[:, None] - 2 * LU + UUq[None, :], LL])
        chi2, U2v = _chi2(U2, W2, q2)
        valid = np.ones((k + 1, k + 1), bool)
        valid[1:, :k] = np.arange(k)[:, None] <= np.arange(k)[None, :]
        valid[0, k] = False                                      # (-inf, +inf) = scope x non-missing
        chi2 = np.where(valid, chi2, 0.0)
        flat = np.argsort(-chi2, axis=None)[:TOP_PER_BLOCK]
        for f in flat:
            a, b = divmod(int(f), k + 1)
            if chi2[a, b] <= 0:
                break
            out.append((chi2[a, b], U2v[a, b], {'type': 'band', 'col': name, 'lo': None if a == 0 else float(grid[a - 1]),
                                                'hi': None if b == k else float(grid[b]), 'scope': s}))
        if (~nn).any():
            B = ((~nn) * m)[:, None].astype(float)
            U, WS, A = ctx.parts(B)
            c, u = _chi2(U, WS, np.einsum('pi,pi->i', A, ctx.Q @ A))
            out.append((c[0], u[0], {'type': 'missing', 'col': name, 'scope': s}))
    return out


def scan_subsets(ctx, z, masks, name, max_levels=12):
    levels = sorted(set(z))
    D = np.column_stack([(z == lv).astype(float) for lv in levels])
    k = len(levels)
    if k <= max_levels:
        subs = [s for r in range(1, k) for s in itertools.combinations(range(k), r)]
    else:
        subs = [(i,) for i in range(k)]
    S = np.zeros((k, len(subs)))
    for j, s in enumerate(subs):
        S[list(s), j] = 1
    out = []
    for sc, m in masks.items():
        Dm = D * m[:, None]
        UL, WL, AL = ctx.parts(Dm)
        U, WS, A = UL @ S, WL @ S, AL @ S
        chi, u = _chi2(U, WS, np.einsum('pi,pi->i', A, ctx.Q @ A))
        for j in [j for j in np.argsort(-chi)[:TOP_PER_BLOCK] if chi[j] > 0]:
            out.append((chi[j], u[j], {'type': 'subset', 'col': name, 'levels': [levels[i] for i in subs[j]], 'scope': sc}))
    return out


def pair_indicators(num, cat, base, ncut=6):
    """For each base column, a small set of indicators: x >= c and x < c at `ncut` quantile cuts (numeric), or levels."""
    ind = {}
    for c in base:
        if c in cat:
            z = cat[c]
            for lv in sorted(set(z)):
                if (z == lv).mean() >= 0.01:
                    ind[(c, '=', lv)] = (z == lv).astype(float)
        else:
            x = num[c]
            g = cut_grid(x, ncut)
            for th in g:
                ind[(c, '>=', float(th))] = np.nan_to_num(x >= th).astype(float) * ~np.isnan(x)
                ind[(c, '<', float(th))] = np.nan_to_num(x < th).astype(float) * ~np.isnan(x)
    return ind


def scan_pairs(ctx, ind, masks):
    keys = list(ind)
    cols = np.array([k[0] for k in keys])
    M = np.column_stack([ind[k] for k in keys])                  # n x K
    out = []
    p = ctx.XW.shape[1]
    for sc, m in masks.items():
        Mm = M * m[:, None]
        for c in dict.fromkeys(cols):
            ii = np.nonzero(cols == c)[0]
            jj = np.nonzero(cols > c)[0]                         # each unordered column pair once
            if len(jj) == 0:
                continue
            Bi, Bj = Mm[:, ii], M[:, jj]
            U = (Bi * ctx.r[:, None]).T @ Bj
            WS = (Bi * ctx.W[:, None]).T @ Bj
            A = np.stack([(Bi * ctx.XW[:, q:q + 1]).T @ Bj for q in range(p)])     # p x |ii| x |jj|
            quad = np.einsum('pab,pab->ab', A, np.einsum('pq,qab->pab', ctx.Q, A))
            chi, u = _chi2(U, WS, quad)
            ia, ib = np.nonzero(chi > 0)
            top = np.argsort(-chi[ia, ib])[:50]                  # keep the best 50 per column pair-block and scope
            for t in top:
                a, b = ia[t], ib[t]
                out.append((chi[a, b], u[a, b], {'type': 'pair', 'a': list(keys[ii[a]]), 'b': list(keys[jj[b]]), 'scope': sc}))
    return out


def scan(ctx, num, cat, d, families=('band', 'subset', 'pair'), pair_base=None, keep=300):
    """Run all families; return (top-`keep` candidates as a DataFrame, max chi2 per family)."""
    masks = {s: scope_mask(d, s) for s in SCOPES}
    res, mx = [], {}
    if 'band' in families:
        b = []
        for name, x in num.items():
            g = cut_grid(x)
            if len(g):
                b += scan_bands(ctx, x, masks, name, g)
        mx['band'] = max(r[0] for r in b); res += sorted(b, key=lambda r: -r[0])[:keep]
    if 'subset' in families:
        s = []
        for name, z in cat.items():
            if len(set(z)) > 1:
                s += scan_subsets(ctx, z, masks, name)
        mx['subset'] = max(r[0] for r in s); res += sorted(s, key=lambda r: -r[0])[:keep]
    if 'pair' in families and pair_base:
        pm = {k: v for k, v in masks.items() if k in PAIR_SCOPES}
        pr = scan_pairs(ctx, pair_indicators(num, cat, pair_base), pm)
        mx['pair'] = max(r[0] for r in pr); res += sorted(pr, key=lambda r: -r[0])[:keep]
    df = pd.DataFrame([{'chi2': c, 'direction': '+' if u > 0 else '-', 'family': r['type'] if r['type'] != 'missing' else 'band',
                        'rule': r} for c, u, r in res]).sort_values('chi2', ascending=False).reset_index(drop=True)
    df['label'] = df.rule.map(describe)
    mx['global'] = max(mx.values())
    return df, mx


# ---------------------------------------------------------------- rules -> columns
def materialize(rule, num, cat, d):
    m = scope_mask(d, rule['scope'])
    t = rule['type']
    if t == 'band':
        x = num[rule['col']]
        v = ~np.isnan(x)
        if rule['lo'] is not None:
            v &= np.nan_to_num(x, nan=-np.inf) >= rule['lo']
        if rule['hi'] is not None:
            v &= np.nan_to_num(x, nan=np.inf) <= rule['hi']
        return v.astype(float) * m
    if t == 'missing':
        return np.isnan(num[rule['col']]).astype(float) * m
    if t == 'subset':
        return np.isin(cat[rule['col']], rule['levels']).astype(float) * m
    if t == 'pair':
        def one(key):
            c, op, v = key
            if op == '=':
                return (cat[c] == v).astype(float)
            x = num[c]
            return ((x >= v) if op == '>=' else (x < v)).astype(float) * ~np.isnan(x)
        return one(rule['a']) * one(rule['b']) * m
    raise ValueError(t)


def describe(r):
    sc = '' if r['scope'] == 'all' else f" @{r['scope']}"
    t = r['type']
    if t == 'band':
        lo, hi = r['lo'], r['hi']
        cond = f'{r["col"]} >= {lo:g}' if hi is None else (f'{r["col"]} <= {hi:g}' if lo is None else f'{lo:g} <= {r["col"]} <= {hi:g}')
        return cond + sc
    if t == 'missing':
        return f'{r["col"]} missing' + sc
    if t == 'subset':
        return f'{r["col"]} in {r["levels"]}' + sc
    return f'[{" ".join(map(str, r["a"]))}] & [{" ".join(map(str, r["b"]))}]' + sc

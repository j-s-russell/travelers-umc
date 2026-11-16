"""Non-linear effect tests against SPEC8 (19_nonlinear).
A  functional form of SPEC8's linear / ordinal terms: refit with each alternative (Box-Cox grid, x + x^2, splines, hinges, bins,
   one-hot), likelihood-ratio gain -> p-value with the extra degrees of freedom.
B  smooth main effects (k = 4 spline blocks, one-hot for <= 8 levels) of columns outside / other scopes: multi-df score test.
C  smooth pairwise interactions (centred 3x3 spline-product blocks) and price ratios: score test.
D  link function: Pregibon eta^2 and Stukel's two generalized-logistic terms: score test.
Family statistic = max -log10 p over its candidates; pure-chance threshold = 95th percentile over outcome sets simulated from the
fitted SPEC8 + N(0, 0.16) ZIP3 2017-18 factors (as final_check.py). E (power) injects known non-linear truths into the simulations.
Run from models/: nonlinear_screen.py N_NULL N_POWER OUT"""
import sys, pickle, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.preprocessing import SplineTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import formula as FM
import formula_steps as FS

L, y = FS.L, FS.y.astype(float)
SPEC = FS.SPEC8
ERA = L.ERA.astype(str).to_numpy()
N = len(y)
MASKS = {'all': np.ones(N), 'early': (ERA != '2017-18').astype(float), '2015': (ERA == '2015').astype(float),
         '2016': (ERA == '2016').astype(float), 'late': (ERA == '2017-18').astype(float)}
SCOPE_MASKS = {'all': ['all'], 'early': ['early'], 'late': ['late'], '2015': ['2015'], '2016': ['2016'],
               'split': ['early', 'late'], 'per-era': ['2015', '2016', 'late']}
TAU_SIM = 0.16
Z3 = L.ZIP_CD.astype(str).str[:3].to_numpy()


# ---------------------------------------------------------------- design and fitting
def term_cols(t, d=L):
    f = FM.Formula([t], 1, '2era'); f.state = {}
    return f._design(d, fit=True)[:, 1:]                       # drop the late intercept (kept once in the base)


def base_design():
    blocks, slices, j = [np.ones((N, 1)), MASKS['late'][:, None]], {}, 2
    for i, t in enumerate(SPEC):
        c = term_cols(t); blocks.append(c); slices[i] = slice(j, j + c.shape[1]); j += c.shape[1]
    return np.hstack(blocks), slices


def _std(X):
    s = X.std(0); s[s == 0] = 1
    m = X.mean(0); m[0] = 0
    return (X - m) / s


def irls(X, yy, it=50):
    X = _std(X)
    b = np.zeros(X.shape[1]); b[0] = np.log(yy.mean() / (1 - yy.mean()))
    for _ in range(it):
        eta = X @ b; p = 1 / (1 + np.exp(-eta)); w = p * (1 - p)
        H = X.T @ (X * w[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (yy - p))
        b += step
        if np.abs(step).max() < 1e-7:
            break
    eta = X @ b; p = 1 / (1 + np.exp(-eta))
    dev = -2 * np.sum(yy * np.log(np.clip(p, 1e-15, 1)) + (1 - yy) * np.log(np.clip(1 - p, 1e-15, 1)))
    return eta, p, dev


X0, SLICES = base_design()


# ---------------------------------------------------------------- A: functional-form variants
def _fill(col):
    x = L[col].astype(float)
    return x.fillna(x.median()).to_numpy()


def _bc(x, lam):
    x = x / np.median(x)
    return np.log(x) if lam == 0 else (x ** lam - 1) / lam


def _spl(x, k, drop=True):
    B = SplineTransformer(n_knots=k, degree=3, knots='quantile', extrapolation='constant').fit_transform(x[:, None])
    return B[:, 1:] if drop else B


def _onehot_prob(col, levels):
    """one-hot of a discrete column; missing rows get the level shares (missing was masked after the outcome)."""
    v = L[col]
    cols = np.column_stack([(v == lv).to_numpy(float) for lv in levels])
    share = cols[v.notna().to_numpy()].mean(0)
    cols[v.isna().to_numpy()] = share
    return cols[:, 1:]


BC_GRID = (-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.5, 2.0)


def variants(col):
    """name -> unscoped base columns, for each alternative form of SPEC8 term `col`."""
    if col in ('QUOTED_AMT', 'DEN_KM2'):
        x = _fill(col)
        q1, q2 = np.quantile(x, [1 / 3, 2 / 3])
        bins = np.digitize(x, np.quantile(x, [.2, .4, .6, .8]))
        out = {f'boxcox {lam:g}': _bc(x, lam)[:, None] for lam in BC_GRID}
        out |= {'x + x^2': np.column_stack([x, x ** 2]), 'spline k4': _spl(x, 4), 'spline k6': _spl(x, 6),
                'hinge terciles': np.column_stack([x, np.maximum(0, x - q1), np.maximum(0, x - q2)]),
                'quintile bins': np.column_stack([(bins == j).astype(float) for j in range(1, 5)])}
        return out
    if col == 'NUM_DRIVERS':
        x = _fill(col)
        return {'one-hot (1,2,3,4,5+)': np.column_stack([(np.minimum(x, 5) == v).astype(float) for v in (2, 3, 4, 5)]),
                'capped at 3': np.minimum(x, 3)[:, None], 'capped at 4': np.minimum(x, 4)[:, None], 'log': np.log(x)[:, None],
                'x + x^2': np.column_stack([x, x ** 2])}
    if col == 'CAT_ZONE':
        x = _fill(col)
        return {'one-hot': _onehot_prob(col, [3.0, 1.0, 2.0, 4.0, 5.0]), 'x + x^2': np.column_stack([x, x ** 2]),
                'log': np.log(x)[:, None], 'sqrt': np.sqrt(x)[:, None]}
    if col == 'PKG_ORDINAL_IMP':
        v = L.COV_PKG_TYPE.astype(str).where(L.COV_PKG_TYPE.astype(str) != 'Missing')
        Lp = pd.DataFrame({'PKG': v.where(v.isin(['Low', 'Medium', 'High']))})
        cols = np.column_stack([(Lp.PKG == lv).to_numpy(float) for lv in ['Medium', 'Low', 'High']])
        miss = Lp.PKG.isna().to_numpy(); cols[miss] = cols[~miss].mean(0)
        return {'one-hot': cols[:, 1:]}
    raise ValueError(col)


A_TERMS = [i for i, t in enumerate(SPEC) if t['col'] in ('QUOTED_AMT', 'DEN_KM2', 'NUM_DRIVERS', 'CAT_ZONE', 'PKG_ORDINAL_IMP')]
A_DESIGNS = {}
for i in A_TERMS:
    t = SPEC[i]; keep = np.ones(X0.shape[1], bool); keep[SLICES[i]] = False
    for name, B in variants(t['col']).items():
        Xv = np.hstack([X0[:, keep]] + [B * MASKS[m][:, None] for m in SCOPE_MASKS[t['scope']]])
        A_DESIGNS[(t['col'], name)] = (Xv, Xv.shape[1] - X0.shape[1])


def boxcox_profile(col, yy, lams=np.round(np.arange(-1.5, 2.01, 0.25), 2)):
    i = next(j for j, t in enumerate(SPEC) if t['col'] == col)
    keep = np.ones(X0.shape[1], bool); keep[SLICES[i]] = False
    x = _fill(col)
    dev = [irls(np.hstack([X0[:, keep]] + [_bc(x, l)[:, None] * MASKS[m][:, None] for m in SCOPE_MASKS[SPEC[i]['scope']]]), yy)[2]
           for l in lams]
    return pd.Series(dev, index=lams)


# ---------------------------------------------------------------- B / C / D: score-test blocks
def _basis(x, k=4):
    x = pd.Series(x, dtype=float); x = x.fillna(x.median()).to_numpy()
    u = np.unique(x)
    if len(u) <= 8:
        vc = pd.Series(x).value_counts()
        return np.column_stack([(x == v).astype(float) for v in vc.index[1:]])
    return _spl(x, k)


def _centered(x, k=3):
    B = _basis(x, k)
    return (B - B.mean(0))[:, : min(B.shape[1], 3)]


def _expected_log_price():
    from impute import _price_model
    import lightgbm as lgb
    d = L.copy()
    _, feats = _price_model(d)
    X = d[feats].copy()
    for c in X:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype('category')
    ok = d.QUOTED_AMT.notna().to_numpy(); out = np.full(N, np.nan)
    for tr, va in KFold(5, shuffle=True, random_state=0).split(X):
        tr = tr[ok[tr]]
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1).fit(X.iloc[tr], np.log(d.QUOTED_AMT.iloc[tr]))
        out[va] = m.predict(X.iloc[va])
    return out


def main_candidates():
    dt = pd.to_datetime(L.QUOTE_DT)
    start = pd.to_datetime(np.where(ERA == '2015', '2015-01-01', np.where(ERA == '2016', '2016-01-01', '2017-01-01')))
    price = L.QUOTED_AMT.astype(float)
    cols = {c: L[c].astype(float).to_numpy() for c in
            ['CREDIT_SCORE', 'QUOTED_AMT', 'DEN_KM2', 'ACS_POP_DENSITY', 'DRV_AGE_MEAN', 'DRV_AGE_MIN', 'DRV_AGE_MAX', 'VEH_MILEAGE_SUM',
             'VEH_MILEAGE_MAX', 'DRV_COMMUTE_SUM', 'DRV_COMMUTE_MAX', 'DRV_ACC_FREE_MIN', 'DRV_ACC_FREE_MEAN', 'VEH_AGE_MIN', 'VEH_AGE_MEAN',
             'VEH_AGE_MAX', 'DRV_SAFETY_MIN', 'DRV_SAFETY_MEAN', 'VEH_SAFETY_MIN', 'VEH_SAFETY_MEAN', 'DAYS_TO_FOLLOWUP', 'PRIOR_CLAIMS_CNT',
             'DRIVERS_PER_VEH', 'NUM_DRIVERS', 'VEH_N_ROWS', 'CAT_ZONE', 'ACS_MED_HH_INCOME']}
    cols['DAY_IN_ERA'] = (dt - start).dt.days.astype(float).to_numpy()
    cols['log price / vehicle'] = np.log(price / L.VEH_N_ROWS.astype(float)).to_numpy()
    cols['log price / driver'] = np.log(price / L.NUM_DRIVERS.astype(float)).to_numpy()
    cols['log price / state median'] = np.log(price / price.groupby(L.STATE_CD.astype(str)).transform('median')).to_numpy()
    cols['log price / expected (OOF)'] = np.log(price.to_numpy()) - _expected_log_price()
    cols['log price / credit'] = np.log(price / L.CREDIT_SCORE.astype(float)).to_numpy()
    return {k: _basis(v) for k, v in cols.items()}


PAIRS = [('QUOTED_AMT', b) for b in ['CREDIT_SCORE', 'NUM_DRIVERS', 'VEH_N_ROWS', 'PKG_ORDINAL_IMP', 'CAT_ZONE', 'DEN_KM2',
                                     'VEH_LUXURY_SHARE', 'DRV_AGE_MEAN', 'DRIVERS_PER_VEH', 'DRV_ACC_FREE_MIN']] + \
        [('DEN_KM2', b) for b in ['R_EDU_MAJ_P', 'DRV_AGE_MEAN', 'NUM_DRIVERS', 'CREDIT_SCORE', 'CAT_ZONE', 'PKG_ORDINAL_IMP']] + \
        [('CREDIT_SCORE', b) for b in ['NUM_DRIVERS', 'DRV_AGE_MEAN', 'PKG_ORDINAL_IMP', 'CAT_ZONE', 'DRIVERS_PER_VEH']] + \
        [('NUM_DRIVERS', 'DRV_AGE_MEAN'), ('DRV_AGE_MEAN', 'CAT_ZONE'), ('DRV_AGE_MEAN', 'PKG_ORDINAL_IMP'), ('CAT_ZONE', 'PKG_ORDINAL_IMP'),
         ('VEH_MIN_GE5_P', 'DRV_ACC_FREE_MIN'), ('VEH_MILEAGE_SUM', 'DRV_COMMUTE_SUM')]


def pair_candidates():
    out = {}
    for a, b in PAIRS:
        A, B = _centered(L[a].to_numpy(float)), _centered(L[b].to_numpy(float))
        out[f'{a} x {b}'] = np.column_stack([A[:, i] * B[:, j] for i in range(A.shape[1]) for j in range(B.shape[1])])
    return out


B_CANDS, C_CANDS = main_candidates(), pair_candidates()
SCOPES_BC = ['all', 'early', '2015', '2016', 'late']


def score_block(Xb, Xs, p, yy, Minv):
    w = p * (1 - p)
    s = Xb.std(0); keep = s > 0
    if not keep.any():
        return 0.0, 0
    Xb = Xb[:, keep] / s[keep]
    U = Xb.T @ (yy - p)
    Bm = Xs.T @ (Xb * w[:, None])
    I = Xb.T @ (Xb * w[:, None]) - Bm.T @ Minv @ Bm
    v, Q = np.linalg.eigh((I + I.T) / 2)
    ok = v > max(1e-4 * v.max(), 1.0)
    if not ok.any():
        return 0.0, 0
    return float(((Q.T @ U)[ok] ** 2 / v[ok]).sum()), int(ok.sum())


def _mlp(chi, df):
    return -np.log10(max(chi2.sf(chi, max(df, 1)), 1e-300))


# ---------------------------------------------------------------- one full analysis
def analyze(yy, detail=True):
    eta, p, dev0 = irls(X0, yy)
    A = []
    for (col, name), (Xv, ddf) in A_DESIGNS.items():
        g = dev0 - irls(Xv, yy)[2]
        A.append({'term': col, 'variant': name, 'LR': g, 'extra df': ddf, '-log10 p': _mlp(max(g, 0), ddf)})
    Xs = _std(X0); w = p * (1 - p)
    Minv = np.linalg.pinv(Xs.T @ (Xs * w[:, None]))
    Bres, Cres = [], []
    for res, cands in ((Bres, B_CANDS), (Cres, C_CANDS)):
        for name, Xb in cands.items():
            for sc in SCOPES_BC:
                chi, df = score_block(Xb * MASKS[sc][:, None], Xs, p, yy, Minv)
                res.append({'candidate': name, 'scope': sc, 'chi2': chi, 'df': df, '-log10 p': _mlp(chi, df) if df else 0.0})
    e = eta
    Dres = []
    for name, Xb in {'Pregibon eta^2': (e ** 2)[:, None],
                     'Stukel (2 terms)': np.column_stack([0.5 * e ** 2 * (e >= 0), -0.5 * e ** 2 * (e < 0)])}.items():
        chi, df = score_block(Xb, Xs, p, yy, Minv)
        Dres.append({'test': name, 'chi2': chi, 'df': df, '-log10 p': _mlp(chi, df)})
    out = {'A': pd.DataFrame(A), 'B': pd.DataFrame(Bres), 'C': pd.DataFrame(Cres), 'D': pd.DataFrame(Dres), 'dev0': dev0}
    out['family'] = {k: float(out[k]['-log10 p'].max()) for k in 'ABCD'}
    out['A term'] = out['A'].groupby('term')['-log10 p'].max().to_dict()
    if not detail:
        out = {k: out[k] for k in ('family', 'A term')} | {'B max': out['B'].groupby('candidate')['-log10 p'].max().to_dict(),
                                                           'C max': out['C'].groupby('candidate')['-log10 p'].max().to_dict()}
    return out


# ---------------------------------------------------------------- simulations
def truth():
    eta, _, _ = irls(X0, y)
    return eta


def _resid(g, w):
    """component of g orthogonal (weighted) to the base design."""
    Xs = _std(X0)
    b = np.linalg.lstsq(Xs * np.sqrt(w)[:, None], g * np.sqrt(w), rcond=None)[0]
    r = g - Xs @ b
    return r / r.std()


def injections(eta0):
    p0 = 1 / (1 + np.exp(-eta0)); w = p0 * (1 - p0)
    pr, den, cr = _fill('QUOTED_AMT'), _fill('DEN_KM2'), _fill('CREDIT_SCORE')
    return {'log price (per era)': _resid(np.column_stack([np.log(pr) * MASKS[m] for m in ('2015', '2016', 'late')]).sum(1), w),
            'sqrt density (late)': _resid(np.sqrt(den) * MASKS['late'], w),
            'credit curve (late)': _resid(-((cr - 650) / 100) ** 2 * MASKS['late'], w),
            'price x credit': _resid((pr - pr.mean()) / pr.std() * (cr - cr.mean()) / cr.std(), w)}


if __name__ == '__main__':
    N_NULL, N_POW, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    t0 = time.time()
    res = {'real': analyze(y)}
    res['real']['boxcox'] = {c: boxcox_profile(c, y) for c in ('QUOTED_AMT', 'DEN_KM2')}
    print('real', res['real']['family'], f'{time.time() - t0:.0f}s', flush=True)
    eta0 = truth()
    codes, inv = np.unique(Z3, return_inverse=True)
    rng = np.random.default_rng(2028)
    def draw(extra=0.0):
        e = eta0 + np.where(MASKS['late'] > 0, rng.normal(0, TAU_SIM, len(codes))[inv], 0) + extra
        pt = 1 / (1 + np.exp(-e))
        return (rng.random(N) < pt).astype(float), pt, e
    res['null'] = []
    for i in range(N_NULL):
        ys, _, _ = draw()
        res['null'].append(analyze(ys, detail=False))
        if i % 10 == 9:
            print('null', i + 1, f'{time.time() - t0:.0f}s', flush=True); pickle.dump(res, open(OUT, 'wb'))
    res['power'] = []
    for name, g in injections(eta0).items():
        for sd in (0.1, 0.2, 0.3):
            for j in range(N_POW):
                ys, pt, e = draw(sd * g)
                r = analyze(ys, detail=False)
                r |= {'injection': name, 'sd': sd, 'oracle gain': roc_auc_score(ys, pt) - roc_auc_score(ys, e - sd * g)}
                res['power'].append(r)
            print('power', name, sd, f'{time.time() - t0:.0f}s', flush=True); pickle.dump(res, open(OUT, 'wb'))
    pickle.dump(res, open(OUT, 'wb'))

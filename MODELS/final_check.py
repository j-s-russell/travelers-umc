"""Simulation test + residual-CatBoost depth test for the final model (SPEC8 + shrunken ZIP3 2017-18 adjustment), seed-0 folds.
Truth for simulations: SPEC8 fitted on all labeled quotes, plus a fresh N(0, TAU) territory factor per ZIP3 in 2017-18.
Per run: final model, CatBoost (07), blend 0.9/0.1, oracle (sims), residual CatBoost of depth 1/2/3/4/6 on the final model's log-odds.
Run from models/: final_check.py N_SIM OUT"""
import sys, pickle, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import catboost as cbm
from sklearn.metrics import roc_auc_score, log_loss

import common as C, features as FE, formula as FM, formula_steps as FS

N_SIM, OUT = int(sys.argv[1]), sys.argv[2]
TAU2, TAU_SIM, DEPTHS, TREES = 0.03, 0.16, (1, 2, 3, 4, 6), 800
L, y, lab = FS.L, FS.y, FS.lab
late = (L.ERA.astype(str) == '2017-18').to_numpy()
z3 = L.ZIP_CD.astype(str).str[:3].to_numpy()
FOLDS0 = [f for f in C.folds(lab) if f[0] == 0]
PC = json.load(open('params_cats.json'))
LC = FE.add_cats(FE.add_static(lab))
# broad feature set for the residual model (as in resid_depth.py)
D = FE.add_era(FE.add_cats(FE.add_static(lab)))
D['QUOTE_DAYNUM'] = (pd.to_datetime(D.QUOTE_DT) - pd.Timestamp('2015-01-01')).dt.days
for c in ('DEN_KM2', 'R_EDU_MAJ_P', 'VEH_MIN_GE5_P', 'PKG_ORDINAL'):
    D[c] = L[c].to_numpy()
D['ZIP3'] = z3
drop = {'SPLIT', 'QUOTE_DT', 'ZIP_CD', 'POLICY_ID', 'CONVERT_IND', 'CARRIER_STATE', 'PKG_CAT', 'C_DRV_VEH',
        'C_LIVING_PROFILE', 'C_MARITAL_PROFILE', 'C_AGE_PROFILE', 'C_GENDER_PROFILE'}
RX = [c for c in D.columns if c not in drop]
rcats = [c for c in RX if not pd.api.types.is_numeric_dtype(D[c])]
XR = C.cb_X(D[RX], rcats)


def fit_final(Ltr, ytr, ztr, ltr):
    m = FM.Formula(FS.SPEC8, 0.1, '2era').fit(Ltr, ytr)
    p = m.predict(Ltr)
    r = pd.DataFrame({'g': ztr[ltr], 'r': (ytr - p)[ltr], 'w': (p * (1 - p))[ltr]}).groupby('g').sum()
    return m, r.r / (r.w + 1 / TAU2)


def decision(m, eff, Ld, zd, ld):
    return m.decision(Ld) + np.where(ld, pd.Series(zd).map(eff).fillna(0).to_numpy(), 0)


def run(yy, tag, p_true=None):
    Lf, Lc = L.assign(CONVERT_IND=yy), LC.assign(CONVERT_IND=yy)
    cb = C.cv_score(f'{tag}_cb', Lc, PC['features'], 'cb', PC['params'], seeds=(0,), log=False)
    Pc = cb['oof_all'][0]
    out = {'final': [], 'CatBoost': [], 'blend 0.9/0.1': [], 'oracle': []}
    base = []
    for s, k, tr, va in FOLDS0:
        m, eff = fit_final(Lf.iloc[tr], yy[tr], z3[tr], late[tr])
        btr, bva = decision(m, eff, Lf.iloc[tr], z3[tr], late[tr]), decision(m, eff, Lf.iloc[va], z3[va], late[va])
        base.append((btr, bva))
        pf = 1 / (1 + np.exp(-bva))
        out['final'].append(roc_auc_score(yy[va], pf)); out['CatBoost'].append(roc_auc_score(yy[va], Pc[va]))
        out['blend 0.9/0.1'].append(roc_auc_score(yy[va], 0.9 * pf + 0.1 * Pc[va]))
        out['oracle'].append(roc_auc_score(yy[va], p_true[va]) if p_true is not None else np.nan)
    res = {k: float(np.mean(v)) for k, v in out.items()} | {'CatBoost trees': cb['rounds']}
    f_ll = np.mean([log_loss(yy[va], 1 / (1 + np.exp(-b[1]))) for (_, _, _, va), b in zip(FOLDS0, base)])
    for d in DEPTHS:
        A, LL = [], []
        for (s, k, tr, va), (btr, bva) in zip(FOLDS0, base):
            r = cbm.CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', custom_metric=['Logloss'], learning_rate=0.03, depth=d,
                                       l2_leaf_reg=9, border_count=32, one_hot_max_size=10, random_seed=0, logging_level='Silent',
                                       allow_writing_files=False, iterations=TREES)
            r.fit(cbm.Pool(XR.iloc[tr], yy[tr], cat_features=rcats, baseline=btr), eval_set=cbm.Pool(XR.iloc[va], yy[va], cat_features=rcats, baseline=bva))
            ev = r.get_evals_result()['validation']; A.append(ev['AUC']); LL.append(ev['Logloss'])
        a = np.mean(A, 0) - res['final']; l = (f_ll - np.mean(LL, 0)) * 1e4
        res[f'resid d{d} AUC'] = float(a.max()); res[f'resid d{d} ll'] = float(l.max()); res[f'resid d{d} trees'] = int(l.argmax() + 1)
    return res


m0, eff0 = fit_final(L, y, z3, late)
p_formula = m0.decision(L)
rows = {'real': run(y, 'final_real')}
print('real', {k: round(v, 5) for k, v in rows['real'].items()}, flush=True)
rng = np.random.default_rng(2027)
codes, inv = np.unique(z3, return_inverse=True)
for i in range(N_SIM):
    b = rng.normal(0, TAU_SIM, len(codes))
    pt = 1 / (1 + np.exp(-(p_formula + np.where(late, b[inv], 0))))
    ys = (rng.random(len(y)) < pt).astype(float)
    rows[f'sim {i + 1}'] = run(ys, f'final_sim{i + 1}', pt)
    print(f'sim {i + 1}', {k: round(v, 5) for k, v in rows[f"sim {i + 1}"].items()}, flush=True)
    pickle.dump(rows, open(OUT, 'wb'))
pickle.dump(rows, open(OUT, 'wb'))

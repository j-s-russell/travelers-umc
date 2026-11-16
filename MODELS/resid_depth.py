"""Interaction order of the leftover signal: residual CatBoost of depth d on top of a formula's log-odds (seed-0 folds).
depth 1 = additive only, 2 = up to pairs, 3 = up to 3-way, ... Real labels vs outcomes simulated from the same formula.
Run from models/: resid_depth.py SPEC_NAME N_SIM OUT"""
import sys, pickle, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import numpy as np, pandas as pd, catboost as cbm
from sklearn.metrics import roc_auc_score, log_loss
import common as C, features as FE, formula as FM, formula_steps as FS
from formula_steps import t

SPECS = {'SPEC6': FS.SPEC6,
         'SPEC7': FS.SPEC6 + [t('DRV_ACC_FREE_MIN', 'step', 'all', cuts=[3]), t('VEH_AGE_MIN', 'step', 'all', cuts=[5])]}
NAME, N_SIM, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SPEC = SPECS[NAME]
DEPTHS, TREES = (1, 2, 3, 4, 6), 800
L, y, lab = FS.L, FS.y, FS.lab
D = FE.add_era(FE.add_cats(FE.add_static(lab)))
D['QUOTE_DAYNUM'] = (pd.to_datetime(D.QUOTE_DT) - pd.Timestamp('2015-01-01')).dt.days
D['PKG_ORDINAL'] = L['PKG_ORDINAL'].to_numpy()
drop = {'SPLIT', 'QUOTE_DT', 'ZIP_CD', 'POLICY_ID', 'CONVERT_IND', 'ZIP3', 'CARRIER_STATE', 'PKG_CAT', 'C_DRV_VEH',
        'C_LIVING_PROFILE', 'C_MARITAL_PROFILE', 'C_AGE_PROFILE', 'C_GENDER_PROFILE'}
FX = [c for c in D.columns if c not in drop]
cats = [c for c in FX if not pd.api.types.is_numeric_dtype(D[c])]
X = C.cb_X(D[FX], cats)
FOLDS0 = [f for f in C.folds(lab) if f[0] == 0]
truth = FM.Formula(SPEC, 0.1, '2era').fit(L, y)
p_star = truth.predict(L)


def run(yy):
    Lf = L.assign(CONVERT_IND=yy)
    out = {}
    base = []
    for s, k, tr, va in FOLDS0:
        m = FM.Formula(SPEC, 0.1, '2era').fit(Lf.iloc[tr], yy[tr])
        base.append((m.decision(Lf.iloc[tr]), m.decision(Lf.iloc[va])))
    f_auc = np.mean([roc_auc_score(yy[va], b[1]) for (_, _, _, va), b in zip(FOLDS0, base)])
    f_ll = np.mean([log_loss(yy[va], 1 / (1 + np.exp(-b[1]))) for (_, _, _, va), b in zip(FOLDS0, base)])
    for d in DEPTHS:
        A, LL = [], []
        for (s, k, tr, va), (btr, bva) in zip(FOLDS0, base):
            r = cbm.CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', custom_metric=['Logloss'], learning_rate=0.03,
                                       depth=d, l2_leaf_reg=9, border_count=32, one_hot_max_size=10, random_seed=0,
                                       logging_level='Silent', allow_writing_files=False, iterations=TREES)
            r.fit(cbm.Pool(X.iloc[tr], yy[tr], cat_features=cats, baseline=btr),
                  eval_set=cbm.Pool(X.iloc[va], yy[va], cat_features=cats, baseline=bva))
            ev = r.get_evals_result()['validation']
            A.append(ev['AUC']); LL.append(ev['Logloss'])
        a, l = np.mean(A, 0) - f_auc, (f_ll - np.mean(LL, 0)) * 1e4      # gains vs formula; ll in 1e-4 units (+ = better)
        out[d] = {'auc_curve': a, 'll_curve': l, 'best_auc': float(a.max()), 'best_ll': float(l.max()), 'best_ll_trees': int(l.argmax() + 1)}
    return out


res = {'spec': NAME, 'real': run(y), 'sims': []}
print('real', {d: (round(v['best_auc'], 5), round(v['best_ll'], 2), v['best_ll_trees']) for d, v in res['real'].items()}, flush=True)
rng = np.random.default_rng(23)
for i in range(N_SIM):
    res['sims'].append(run((rng.random(len(y)) < p_star).astype(float)))
    print(f'sim {i + 1}', {d: (round(v['best_auc'], 5), round(v['best_ll'], 2)) for d, v in res['sims'][-1].items()}, flush=True)
    pickle.dump(res, open(OUT, 'wb'))
pickle.dump(res, open(OUT, 'wb'))

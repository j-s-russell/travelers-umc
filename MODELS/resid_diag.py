"""Where is the leftover signal? Residual CatBoost on the SPEC6 log-odds (fixed trees), SHAP per feature on OOF rows,
real labels vs outcomes simulated from SPEC6 (where any importance is pure noise). Run from models/."""
import sys, pickle, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import numpy as np, pandas as pd, catboost as cbm
from sklearn.metrics import roc_auc_score
import common as C, features as FE, formula as FM, formula_steps as FS

N_SIM, TREES = int(sys.argv[1]), int(sys.argv[2])
OUT = sys.argv[3]
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
P = {'loss_function': 'Logloss', 'eval_metric': 'AUC', 'learning_rate': 0.02, 'depth': 4, 'l2_leaf_reg': 9, 'border_count': 32,
     'one_hot_max_size': 10, 'random_seed': 0, 'logging_level': 'Silent', 'allow_writing_files': False, 'iterations': TREES}
truth = FM.Formula(FS.SPEC6, 0.1, '2era').fit(L, y)
p_star = truth.predict(L)


def run(yy):
    Lf = L.assign(CONVERT_IND=yy)
    shap = np.zeros((len(yy), len(FX)))
    delta = np.zeros(len(yy)); base = np.zeros(len(yy)); curves = []
    for s, k, tr, va in FOLDS0:
        m = FM.Formula(FS.SPEC6, 0.1, '2era').fit(Lf.iloc[tr], yy[tr])
        btr, bva = m.decision(Lf.iloc[tr]), m.decision(Lf.iloc[va])
        r = cbm.CatBoostClassifier(**P)
        pva = cbm.Pool(X.iloc[va], yy[va], cat_features=cats, baseline=bva)
        r.fit(cbm.Pool(X.iloc[tr], yy[tr], cat_features=cats, baseline=btr), eval_set=pva)
        curves.append(np.array(r.get_evals_result()['validation']['AUC']))
        sv = r.get_feature_importance(pva, type='ShapValues')
        shap[va] = sv[:, :-1]
        delta[va] = sv[:, :-1].sum(1); base[va] = bva
    fa = np.mean([roc_auc_score(yy[va], base[va]) for _, _, _, va in FOLDS0])
    return {'shap': shap, 'delta': delta, 'base': base, 'curve': np.mean(curves, 0) - fa, 'formula_auc': fa}


res = {'FX': FX, 'real': run(y), 'sims': []}
print('real done; gain at trees:', {t: round(float(res['real']['curve'][t - 1]), 5) for t in (50, 100, TREES)}, flush=True)
rng = np.random.default_rng(7)
for i in range(N_SIM):
    ys = (rng.random(len(y)) < p_star).astype(float)
    res['sims'].append(run(ys))
    print(f'sim {i + 1} done; gain at {TREES}: {res["sims"][-1]["curve"][-1]:+.5f}', flush=True)
    pickle.dump(res, open(OUT, 'wb'))
pickle.dump(res, open(OUT, 'wb'))

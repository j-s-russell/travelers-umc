"""Pairwise version of resid_diag: residual CatBoost on the SPEC6 log-odds, SHAP interaction values on OOF rows.
For each feature pair, OOF log-likelihood gain of adding only the pair's interaction part (phi_ij + phi_ji) to the formula.
Real labels vs outcomes simulated from SPEC6. Run from models/: resid_inter.py N_SIM TREES OUT"""
import sys, pickle, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import numpy as np, pandas as pd, catboost as cbm
import common as C, features as FE, formula as FM, formula_steps as FS

N_SIM, TREES, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
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
nF = len(FX)
iu = np.triu_indices(nF, 1)


def run(yy):
    Lf = L.assign(CONVERT_IND=yy)
    G = np.zeros(len(iu[0]))            # OOF loglik gain per pair
    M = np.zeros(nF)                    # OOF loglik gain per feature, main part only (phi_ii)
    for s, k, tr, va in FOLDS0:
        m = FM.Formula(FS.SPEC6, 0.1, '2era').fit(Lf.iloc[tr], yy[tr])
        btr, bva = m.decision(Lf.iloc[tr]), m.decision(Lf.iloc[va])
        r = cbm.CatBoostClassifier(**P)
        r.fit(cbm.Pool(X.iloc[tr], yy[tr], cat_features=cats, baseline=btr))
        for a in range(0, len(va), 600):
            idx = va[a:a + 600]
            sv = r.get_feature_importance(cbm.Pool(X.iloc[idx], yy[idx], cat_features=cats, baseline=bva[a:a + 600]),
                                          type='ShapInteractionValues')[:, :nF, :nF]
            b = bva[a:a + 600][:, None]; yv = yy[idx][:, None]
            l0 = -np.logaddexp(0, b)
            phi = (sv + sv.transpose(0, 2, 1))[:, iu[0], iu[1]]
            G += (yv * phi - np.logaddexp(0, b + phi) - l0).sum(0)
            d = sv[:, np.arange(nF), np.arange(nF)]
            M += (yv * d - np.logaddexp(0, b + d) - l0).sum(0)
    return {'pair': G, 'main': M}


res = {'FX': FX, 'iu': iu, 'real': run(y), 'sims': []}
print('real done', flush=True)
rng = np.random.default_rng(11)
for i in range(N_SIM):
    res['sims'].append(run((rng.random(len(y)) < p_star).astype(float)))
    print(f'sim {i + 1} done', flush=True)
    pickle.dump(res, open(OUT, 'wb'))
pickle.dump(res, open(OUT, 'wb'))

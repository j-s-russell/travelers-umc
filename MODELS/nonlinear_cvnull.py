"""CV AUC gain of the strongest Step A variants on outcomes simulated from SPEC8 (+ ZIP3 late factors), seed-0 folds.
Calibrates the real gain of 'density^2 per era' (chosen by the screen) against chance, with and without picking the best of 4.
Run from models/: nonlinear_cvnull.py N_SIM OUT"""
import sys, pickle, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import common as C, formula as FM, formula_steps as FS
from formula_steps import t
import nonlinear_screen as NS

L, y = FS.L, FS.y
L['DEN_KM2_SQ'] = L.DEN_KM2.astype(float) ** 2
F0 = [f for f in C.folds(FS.lab) if f[0] == 0]
V = {'SPEC8': FS.SPEC8, 'density^2 per era': FS.SPEC8 + [t('DEN_KM2_SQ', 'linear', 'per-era')],
     'CAT one-hot': FS.replace(FS.SPEC8, 'CAT_ZONE', [t('CAT_ZONE', 'onehot')]),
     'drivers one-hot': FS.replace(FS.SPEC8, 'NUM_DRIVERS', [t('NUM_DRIVERS_C5', 'onehot', 'split')]),
     'package one-hot': FS.replace(FS.SPEC8, 'PKG_ORDINAL_IMP', [t('COV_PKG_TYPE', 'onehot', 'per-era')])}

def run(yy):
    return {n: np.mean([roc_auc_score(yy[va], FM.Formula(sp, 0.1, '2era').fit(L.iloc[tr], yy[tr]).predict(L.iloc[va])) for _, _, tr, va in F0])
            for n, sp in V.items()}

N_SIM, OUT = int(sys.argv[1]), sys.argv[2]
res = {'real': run(y)}; print('real', res['real'], flush=True)
eta0 = NS.truth(); codes, inv = np.unique(NS.Z3, return_inverse=True); rng = np.random.default_rng(99)
res['sims'] = []
for i in range(N_SIM):
    e = eta0 + np.where(NS.MASKS['late'] > 0, rng.normal(0, NS.TAU_SIM, len(codes))[inv], 0)
    ys = (rng.random(len(y)) < 1 / (1 + np.exp(-e))).astype(float)
    res['sims'].append(run(ys)); pickle.dump(res, open(OUT, 'wb'))
    if i % 5 == 4: print('sim', i + 1, flush=True)

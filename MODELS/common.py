"""Shared modeling harness: data, fixed CV folds, model runners, evaluation, experiment log, submissions.

Every modeling notebook imports this so experiments are compared on identical folds with identical metrics.

Protocol (`cv_score`):
- 5-fold stratified CV x 3 seeds on train+valid (15 fits), folds cached in data/processed/folds.parquet.
- Each fold trains with early stopping. The number of trees is the iteration with the best *mean* AUC
  across all 15 folds, the same rule as `lgb.cv` / `cb.cv` in 01_baselines. Every fold is scored at that iteration.
- An optional `fold_fn(train_part, score_part) -> (train_part, score_part)` hook lets a notebook add target-based
  columns (e.g. target encoding) learned from the training part only, so the scored fold is never seen.
  List the new column names in `feats`.
"""
import hashlib
import json
import pickle
import tempfile
from datetime import datetime
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / 'data' / 'processed'
LOG = ROOT / 'models' / 'experiments.csv'
CACHE = ROOT / 'models' / 'cache'  # cv_score results (git-ignored via *.pkl)
NON_FEATS = ['POLICY_ID', 'SPLIT', 'CONVERT_IND', 'QUOTE_DT', 'ZIP_CD', 'ZIP3']
TARGET = 'CONVERT_IND'

LGB_PARAMS = dict(objective='binary', metric='auc', learning_rate=0.02, num_leaves=15, min_child_samples=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbosity=-1, seed=42)
CB_PARAMS = dict(loss_function='Logloss', eval_metric='AUC', learning_rate=0.03, depth=6, l2_leaf_reg=3, random_seed=42,
                 logging_level='Silent', allow_writing_files=False, train_dir=f'{tempfile.gettempdir()}/catboost_info')
XGB_PARAMS = dict(objective='binary:logistic', eval_metric='auc', tree_method='hist', learning_rate=0.03, max_depth=4,
                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, max_cat_to_onehot=10, seed=42, nthread=-1)
LR_PARAMS = dict(C=0.1, n_knots=5, degree=3)  # spline logistic regression: splines on numerics, one-hot categoricals


# ---------- data ----------
def load():
    """Modeling table split into labeled (train+valid, positional index) and holdout."""
    df = pd.read_parquet(PROC / 'model_table.parquet')
    lab = df[df[TARGET].notna()].reset_index(drop=True)
    hold = df[df.SPLIT == 'holdout'].reset_index(drop=True)
    return lab, hold


def base_features(df):
    return df.columns.drop(NON_FEATS, errors='ignore').tolist()


def folds(lab, n_splits=5, seeds=(0, 1, 2)):
    """Fixed (seed, fold, train_idx, score_idx) splits, cached so every experiment uses the same ones."""
    f = PROC / 'folds.parquet'
    if f.exists():
        t = pd.read_parquet(f)
        assert t.POLICY_ID.tolist() == lab.POLICY_ID.tolist(), 'labeled rows changed; delete folds.parquet'
    else:
        t = lab[['POLICY_ID']].copy()
        for s in seeds:
            t[f'fold_s{s}'] = -1
            for k, (_, va) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=s).split(lab, lab[TARGET])):
                t.iloc[va, t.columns.get_loc(f'fold_s{s}')] = k
        t.to_parquet(f, index=False)
    out = []
    for col in [c for c in t if c.startswith('fold_s')]:
        for k in sorted(t[col].unique()):
            out.append((int(col[6:]), int(k), np.where(t[col] != k)[0], np.where(t[col] == k)[0]))
    return out


# ---------- metrics ----------
def deciles(y, p):
    """Decile table (1 = highest scores): observed rate, mean prediction, lift, cumulative capture."""
    y, p = np.asarray(y), np.asarray(p)
    dec = pd.qcut(pd.Series(p).rank(method='first'), 10, labels=range(10, 0, -1))
    t = pd.DataFrame({'pred': p, 'y': y, 'decile': dec}).groupby('decile', observed=True).agg(
        quotes=('y', 'size'), converted=('y', 'sum'), observed=('y', 'mean'), predicted=('pred', 'mean')).sort_index(ascending=False)
    t['lift'] = t.observed / y.mean()
    t['cum. capture'] = t.converted.cumsum() / t.converted.sum()
    return t


def top_decile_precision(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return y[p >= np.quantile(p, .9)].mean()


# ---------- model runners (one fold) ----------
def cb_X(X, cats):
    """CatBoost needs categoricals as non-null str and numerics as float (no pandas nullable ints)."""
    X = X.astype({c: float for c in X.columns if c not in cats})
    return X.assign(**{c: X[c].astype(object).fillna('Missing').astype(str) for c in cats})


def _run_lgb(Xtr, ytr, Xva, yva, params, max_rounds):
    curve = {}
    m = lgb.train(params, lgb.Dataset(Xtr, ytr), max_rounds, valid_sets=[lgb.Dataset(Xva, yva)],
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.record_evaluation(curve)])
    return m, np.array(curve['valid_0']['auc']), lambda n: m.predict(Xva, num_iteration=n)


def _run_cb(Xtr, ytr, Xva, yva, params, max_rounds):
    cats = Xtr.select_dtypes(['category', 'object', 'string']).columns.tolist()
    m = cb.CatBoostClassifier(**params, iterations=max_rounds, early_stopping_rounds=200, use_best_model=False)
    Xv = cb_X(Xva, cats)
    m.fit(cb_X(Xtr, cats), ytr, cat_features=cats, eval_set=(Xv, yva))
    # a fold that early-stopped before the chosen iteration predicts with all its trees (same as LightGBM)
    return m, np.array(m.get_evals_result()['validation']['AUC']), lambda n: m.predict_proba(Xv, ntree_end=min(n, m.tree_count_))[:, 1]


def xgb_X(X):
    """XGBoost: pandas categoricals stay categorical; everything else float (no nullable ints)."""
    return X.astype({c: float for c in X.columns if not isinstance(X[c].dtype, pd.CategoricalDtype)})


def _run_xgb(Xtr, ytr, Xva, yva, params, max_rounds):
    curve = {}
    dtr, dva = (xgb.DMatrix(xgb_X(a), b, enable_categorical=True) for a, b in ((Xtr, ytr), (Xva, yva)))
    m = xgb.train(params, dtr, max_rounds, evals=[(dva, 'valid')], early_stopping_rounds=200, evals_result=curve, verbose_eval=False)
    return m, np.array(curve['valid']['auc']), lambda n: m.predict(dva, iteration_range=(0, min(n, m.num_boosted_rounds())))


def lr_pipeline(X, params):
    """Spline logistic regression. Continuous numerics (> 10 distinct values): median-impute, cubic B-splines on
    quantile knots. Other numerics (flags, counts): median-impute, linear. Missing indicators for both. Categoricals
    (pandas category): one-hot. A smooth additive model with no interactions, so it errs differently from the GBMs."""
    p = LR_PARAMS | (params or {})
    cats = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    nums = [c for c in X.columns if c not in cats]
    cont = [c for c in nums if X[c].nunique() > 10]
    disc = [c for c in nums if c not in cont]
    spline = make_pipeline(SimpleImputer(strategy='median'),
                           SplineTransformer(n_knots=p['n_knots'], degree=p['degree'], knots='quantile', extrapolation='constant'))
    pre = ColumnTransformer([('spl', spline, cont), ('lin', SimpleImputer(strategy='median'), disc),
                             ('na', MissingIndicator(), nums),
                             ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=20), cats)])
    return make_pipeline(pre, StandardScaler(), LogisticRegression(C=p['C'], max_iter=3000))


def _run_lr(Xtr, ytr, Xva, yva, params, max_rounds):
    m = lr_pipeline(Xtr, params).fit(Xtr.astype({c: float for c in Xtr.select_dtypes('number').columns}), ytr)
    p = m.predict_proba(Xva.astype({c: float for c in Xva.select_dtypes('number').columns}))[:, 1]
    return m, np.array([roc_auc_score(yva, p)]), lambda n: p  # no boosting rounds: a one-point "curve"


EBM_PARAMS = dict(interactions='era', era_col='ERA', random_state=0, n_jobs=-1)  # explainable boosting machine (GA2M)


def ebm_model(X, params):
    """EBM. interactions='era': one main effect per feature plus a (ERA x feature) pair for every feature, i.e. a separate
    additive shape per era. An int adds that many automatically detected pairs instead. `pairs_extra` = extra name pairs."""
    from interpret.glassbox import ExplainableBoostingClassifier
    p = EBM_PARAMS | (params or {})
    cols = list(X.columns)
    inter = p['interactions']
    if inter == 'era':
        e = cols.index(p['era_col'])
        inter = [(e, i) for i in range(len(cols)) if i != e] + [(cols.index(a), cols.index(b)) for a, b in p.get('pairs_extra', [])]
    return ExplainableBoostingClassifier(interactions=inter, random_state=p['random_state'], n_jobs=p['n_jobs'])


def ebm_X(X):
    return X.astype({c: (float if pd.api.types.is_numeric_dtype(X[c]) else str) for c in X.columns})


def _run_ebm(Xtr, ytr, Xva, yva, params, max_rounds):
    m = ebm_model(Xtr, params).fit(ebm_X(Xtr), ytr)
    p = m.predict_proba(ebm_X(Xva))[:, 1]
    return m, np.array([roc_auc_score(yva, p)]), lambda n: p


FORMULA_PARAMS = dict(spec=[], C=1.0)  # explicit logistic formula (models/formula.py); feats = formula.columns(spec)


def _run_formula(Xtr, ytr, Xva, yva, params, max_rounds):
    import formula as FM
    m = FM.Formula(params['spec'], params.get('C', 1.0), params.get('intercepts', '3era')).fit(Xtr, ytr)
    p = m.predict(Xva)
    return m, np.array([roc_auc_score(yva, p)]), lambda n: p


RUNNERS = {'lgb': (_run_lgb, LGB_PARAMS), 'cb': (_run_cb, CB_PARAMS), 'xgb': (_run_xgb, XGB_PARAMS), 'lr': (_run_lr, LR_PARAMS),
           'ebm': (_run_ebm, EBM_PARAMS), 'formula': (_run_formula, FORMULA_PARAMS)}


# ---------- cross-validation ----------
def cv_score(name, lab, feats, model='lgb', params=None, fold_fn=None, max_rounds=5000, log=True, notes='', cache=True, seeds=None,
             rounds=None):
    """Score a feature set / model on the fixed folds. Returns dict with metrics and seed-averaged OOF predictions.

    With `cache`, results are stored in models/cache/ keyed by name + features + model + params (+ fold_fn name),
    and reused on re-runs, so a crash late in a notebook doesn't cost the earlier fits. Fitted models aren't cached.
    `seeds` restricts to some CV repeats (e.g. (0,) = one 5-fold split, for fast hyperparameter search).
    `rounds` fixes the number of trees instead of picking the best mean-curve iteration (e.g. re-running tuned params
    with another model seed). `oof_all` holds one OOF vector per CV repeat, for seed-averaging and blending per fold.
    """
    run, default = RUNNERS[model]
    params = default | (params or {})
    key = hashlib.md5(json.dumps([name, feats, model, params, getattr(fold_fn, '__qualname__', None),
                                  getattr(fold_fn, 'tag', None), len(lab), seeds] + ([rounds] if rounds else []),
                                 default=str).encode()).hexdigest()[:12]
    f = CACHE / f'{name}_{key}.pkl'
    if cache and f.exists():
        return pickle.load(open(f, 'rb'))
    fits = []
    for seed, k, tr, va in folds(lab):
        if seeds is not None and seed not in seeds:
            continue
        a, b = lab.iloc[tr], lab.iloc[va]
        if fold_fn is not None:
            a, b = fold_fn(a, b)
        m, curve, predict = run(a[feats], a[TARGET], b[feats], b[TARGET], params, rounds or max_rounds)
        fits.append((seed, va, curve, predict, m, feats))

    # best iteration = argmax of the mean validation-AUC curve (curves padded with their last value)
    L = max(len(f[2]) for f in fits)
    mean_curve = np.mean([np.pad(f[2], (0, L - len(f[2])), mode='edge') for f in fits], axis=0)
    best = rounds or int(mean_curve.argmax()) + 1

    y = lab[TARGET].to_numpy()
    pos = {s: i for i, s in enumerate(sorted({f[0] for f in fits}))}
    oof = np.full((len(pos), len(lab)), np.nan)
    aucs, lls = [], []
    for seed, va, _, predict, _, _ in fits:
        p = predict(best)
        oof[pos[seed], va] = p
        aucs.append(roc_auc_score(y[va], p))
        lls.append(log_loss(y[va], np.clip(p, 1e-6, 1 - 1e-6)))
    oof_mean = oof.mean(axis=0)
    res = {'name': name, 'model': model, 'n_feats': len(fits[0][5]), 'rounds': best,
           'auc_mean': np.mean(aucs), 'auc_std': np.std(aucs), 'auc_se': np.std(aucs) / np.sqrt(len(aucs)),
           'top_decile_prec': np.mean([top_decile_precision(y, o) for o in oof]),
           'fold_aucs': np.array(aucs), 'fold_ll': np.array(lls), 'oof': oof_mean, 'oof_all': oof, 'models': [f[4] for f in fits], 'feats': fits[0][5]}
    if cache:
        CACHE.mkdir(exist_ok=True)
        pickle.dump({k: v for k, v in res.items() if k != 'models'} | {'models': None}, open(f, 'wb'))
    if log:
        row = {k: v for k, v in res.items() if k not in ('fold_aucs', 'fold_ll', 'oof', 'oof_all', 'models', 'feats')} | {
            'time': datetime.now().strftime('%Y-%m-%d %H:%M'), 'notes': notes, 'params': json.dumps(params, default=str)}
        pd.DataFrame([row]).to_csv(LOG, mode='a', header=not LOG.exists(), index=False)
    return res


def summary(results, baseline=None):
    """Comparison table. With `baseline`, adds the *paired* difference: per-fold AUC minus the baseline's AUC on the
    same fold, averaged over the 15 folds, with its standard error. Fold-to-fold noise cancels, so this is far more
    sensitive than comparing means. Rule of thumb: |Δ| > 2 × se_Δ is a real difference."""
    t = pd.DataFrame([{k: r[k] for k in ('name', 'model', 'n_feats', 'rounds', 'auc_mean', 'auc_se', 'top_decile_prec')}
                      for r in results]).set_index('name')
    if baseline is not None:
        base = next(r for r in results if r['name'] == baseline)['fold_aucs']
        d = {r['name']: r['fold_aucs'] - base for r in results}
        t.insert(4, 'Δ auc', {k: v.mean() for k, v in d.items()})
        t.insert(5, 'se Δ', {k: v.std() / np.sqrt(len(v)) for k, v in d.items()})
    return t


# ---------- final fit & submission ----------
def fit_full(lab, feats, model='lgb', params=None, rounds=None):
    """Refit on all labeled data with a fixed number of trees (use rounds from cv_score, scaled up ~1.2x for 25% more data)."""
    params = RUNNERS[model][1] | (params or {})
    if model == 'lgb':
        m = lgb.train(params, lgb.Dataset(lab[feats], lab[TARGET]), rounds)
        return lambda X: m.predict(X[feats])
    if model == 'xgb':
        m = xgb.train(params, xgb.DMatrix(xgb_X(lab[feats]), lab[TARGET], enable_categorical=True), rounds)
        return lambda X: m.predict(xgb.DMatrix(xgb_X(X[feats]), enable_categorical=True))
    if model == 'formula':
        import formula as FM
        m = FM.Formula(params['spec'], params.get('C', 1.0), params.get('intercepts', '3era')).fit(lab, lab[TARGET])
        return lambda X: m.predict(X)
    if model == 'ebm':
        m = ebm_model(lab[feats], params).fit(ebm_X(lab[feats]), lab[TARGET])
        return lambda X: m.predict_proba(ebm_X(X[feats]))[:, 1]
    if model == 'lr':
        f = lambda X: X[feats].astype({c: float for c in X[feats].select_dtypes('number').columns})
        m = lr_pipeline(lab[feats], params).fit(f(lab), lab[TARGET])
        return lambda X: m.predict_proba(f(X))[:, 1]
    cats = lab[feats].select_dtypes(['category', 'object', 'string']).columns.tolist()
    m = cb.CatBoostClassifier(**params, iterations=rounds).fit(cb_X(lab[feats], cats), lab[TARGET], cat_features=cats)
    return lambda X: m.predict_proba(cb_X(X[feats], cats))[:, 1]


def write_submission(name, hold, preds):
    sub = pd.DataFrame({'POLICY_ID': hold.POLICY_ID.values, 'CONVERT_IND_PRED': preds})
    assert list(sub.columns) == pd.read_csv(ROOT / 'sample_submission.csv', nrows=0).columns.tolist()
    assert len(sub) == 7375 and sub.POLICY_ID.is_unique and sub.CONVERT_IND_PRED.between(0, 1).all()
    out = ROOT / 'submissions' / f'{name}_{datetime.now():%Y%m%d}.csv'
    out.parent.mkdir(exist_ok=True)
    sub.to_csv(out, index=False)
    return out

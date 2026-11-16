"""Formula search path for 12_formula: helpers plus the spec at each step (SPEC166 -> SPEC1 -> SPEC2 -> SPEC3 -> SPEC3c).
Every evaluation goes through common.cv_score (cached), so the notebook re-runs quickly."""
import json, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
import common as C, features as FE, formula as FM
lab, hold = C.load()
L = FM.prep(lab)
y = L[C.TARGET].to_numpy()
F = json.load(open(C.ROOT / 'models/params_tuned.json'))['cb']['features']
SPL = ['CREDIT_SCORE', 'QUOTED_AMT', 'ACS_POP_DENSITY', 'DRV_AGE_MEAN']
ONEHOT = ['STATE_CD', 'PRIOR_CARRIER_GRP', 'COV_PKG_TYPE', 'CAT_ZONE']
ERA_FLAGS = ['R_EDU_MAJ', 'DISCOUNT_IND', 'HOME_POLICY_IND']
SHARED = ['DRV_AGE_35_50', 'R_DPV_GT1', 'R_DPV_HALF_TO_1']
LINEAR = [f for f in F if f not in SPL + ONEHOT + ERA_FLAGS + SHARED + ['NUM_DRIVERS', 'DRIVERS_PER_VEH', 'QUOTE_YEAR']
          and not isinstance(L[f].dtype, pd.CategoricalDtype)]
TF = {'QUOTED_AMT': 'log', 'ACS_POP_DENSITY': 'log1p'}

def t(col, form='linear', scope='all', **kw):
    return {'col': col, 'form': form, 'scope': scope, **kw}

SPEC166 = ([t(c, 'spline', 'per-era', k=5, na=True, **({'tf': TF[c]} if c in TF else {})) for c in SPL]
           + [t(c, 'flag', 'per-era') for c in ERA_FLAGS]
           + [t('NUM_DRIVERS_C5', 'onehot', 'per-era')]
           + [t(c, 'flag') for c in SHARED]
           + [t(c, 'onehot') for c in ONEHOT]
           + [t(c) for c in LINEAR])

def run(spec, Cr=0.003, seeds=(0,), name='fx'):
    return C.cv_score(name, L, FM.columns(spec), 'formula', {'spec': spec, 'C': Cr}, seeds=seeds, log=False)

def cmp(r, b, sl=slice(None)):
    da = r['fold_aucs'][sl] - b['fold_aucs'][sl]; dl = b['fold_ll'][sl] - r['fold_ll'][sl]   # dl > 0: new is better
    n = len(da)
    return {'auc': r['fold_aucs'][sl].mean(), 'Δauc': da.mean(), 'se_auc': da.std() / np.sqrt(n),
            'Δll(+better)': dl.mean() * 1e4, 'se_ll': dl.std() / np.sqrt(n) * 1e4}

def nparams(spec):
    return FM.Formula(spec, 1).fit(L.iloc[:4000], y[:4000]).n_params

CGRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
def best(spec, seeds=(0,)):
    rs = {Cr: run(spec, Cr, seeds) for Cr in CGRID}
    Cb = min(rs, key=lambda c: rs[c]['fold_ll'].mean())
    return rs[Cb] | {'C': Cb}

def replace(spec, col, new_terms):
    return [x for x in spec if x['col'] != col] + list(new_terms)

def table(variants, base_spec):
    b = best(base_spec)
    rows = {'(current)': cmp(b, b) | {'C': b['C'], 'params': nparams(base_spec)}}
    for name, sp in variants.items():
        r = best(sp); rows[name] = cmp(r, b) | {'C': r['C'], 'params': nparams(sp)}
    return pd.DataFrame(rows).T.round(5)

SPEC1 = (
    [t('CREDIT_SCORE', 'linear', 'per-era', na=True), t('CREDIT_SCORE', 'step', 'per-era', cuts=[650])]
    + [t('QUOTED_AMT', 'linear', 'per-era', na=True), t('ACS_POP_DENSITY', 'linear', 'per-era', na=True)]
    + [t(c, 'flag', 'per-era') for c in ERA_FLAGS]
    + [t('NUM_DRIVERS', 'linear', 'per-era')]
    + [t(c, 'flag') for c in SHARED]
    + [t(c, 'onehot') for c in ['STATE_CD', 'PRIOR_CARRIER_GRP', 'COV_PKG_TYPE']] + [t('CAT_ZONE', 'linear', na=True)]
    + [t(c) for c in LINEAR])

def rescope(spec, col, form, scope):
    return [dict(x, scope=scope) if (x['col'] == col and x['form'] == form) else x for x in spec]

SPEC2 = rescope(rescope(rescope(rescope(rescope(rescope(SPEC1, 'CREDIT_SCORE', 'linear', 'all'), 'CREDIT_SCORE', 'step', '2015'),
        'R_EDU_MAJ', 'flag', 'late'), 'DISCOUNT_IND', 'flag', '2016'), 'HOME_POLICY_IND', 'flag', '2016'), 'NUM_DRIVERS', 'linear', 'split')

def drop(spec, cols):
    return [x for x in spec if x['col'] not in cols]

def table_fixed(variants, base_spec, Cr):
    b = run(base_spec, Cr)
    rows = {'(current)': cmp(b, b) | {'params': nparams(base_spec)}}
    for name, sp in variants.items():
        rows[name] = cmp(run(sp, Cr), b) | {'params': nparams(sp)}
    return pd.DataFrame(rows).T.round(5)

DROP3 = ([c for c in LINEAR if c.startswith('ACS_')] + ['STATE_CD', 'DRV_SAFETY_MIN', 'DRV_SAFETY_MEAN', 'DRV_ACC_FREE_MIN', 'DRV_ACC_FREE_MEAN',
         'DRV_OWN_SHARE', 'DRV_RENT_SHARE', 'DRV_HIGH_EDU_SHARE', 'DRV_COMMUTE_SUM', 'DRV_COMMUTE_MAX', 'DRV_AGE_MIN', 'DRV_AGE_MAX',
         'DRV_AGE_SPREAD', 'DRV_YOUNG_SHARE', 'N_YOUNG_DRIVERS', 'QUOTED_AMT_MISSING', 'HOME_POLICY_IND'])
SPEC3 = drop(SPEC2, DROP3)

SPEC3c = rescope(rescope(rescope(drop(SPEC3, ['VEH_N_ROWS']), 'R_DPV_GT1', 'flag', 'per-era'), 'R_DPV_HALF_TO_1', 'flag', 'per-era'),
                 'VEH_LUXURY_SHARE', 'linear', 'split')

# ---------- 13_formula_v2: parsimony pass on SPEC3c (rule definitions in SPEC3c were confirmed optimal by scans) ----------
SPEC4 = [dict(x, na=False) if x.get('na') else x for x in SPEC3c]                                   # no missing flags
SPEC4 = rescope(rescope(SPEC4, 'R_DPV_GT1', 'flag', 'split'), 'R_DPV_HALF_TO_1', 'flag', 'split')   # 2015 = 2016
SPEC4 = rescope(SPEC4, 'VEH_LUXURY_SHARE', 'linear', 'late')
SPEC4 = replace(SPEC4, 'COV_PKG_TYPE', [t('PKG_ORDINAL', 'linear')])
SPEC4 = [x for x in SPEC4 if not (x['col'] == 'CREDIT_SCORE' and x['form'] == 'linear')]           # credit = 2015 step at 650 only
SPEC4_INTERCEPTS = '2era'                                                                          # 2015 and 2016 share an intercept

def run4(spec, Cr=0.1, seeds=(0,), intercepts=SPEC4_INTERCEPTS, name='fx'):
    return C.cv_score(name, L, FM.columns(spec), 'formula', {'spec': spec, 'C': Cr, 'intercepts': intercepts}, seeds=seeds, log=False)


# ---------- 16_rule_scan ----------
SPEC5 = SPEC4 + [t('R_CARRIER_3_8_MISS_2016', 'flag')]          # + the one rule passing the exhaustive scan's pure-chance threshold
SPEC6 = rescope(SPEC5, 'PKG_ORDINAL', 'linear', 'per-era')      # + package effect per era (2016: -0.32 per level)

# ---------- 17_residual_hunt: two steps localized by SHAP of a residual CatBoost (each below the scan's chance threshold alone) ----------
SPEC7 = SPEC6 + [t('DRV_ACC_FREE_MIN', 'step', 'all', cuts=[3]),   # any driver with < 3 accident-free years: +0.25 log-odds
                 t('VEH_AGE_MIN', 'step', 'all', cuts=[5])]        # newest vehicle 5+ years old: -0.18

# ---------- 17_external_sources ----------
SPEC8 = [dict(x, col={'ACS_POP_DENSITY': 'DEN_KM2', 'R_EDU_MAJ': 'R_EDU_MAJ_P', 'PKG_ORDINAL': 'PKG_ORDINAL_IMP'}.get(x['col'], x['col']))
         for x in SPEC7 if x['col'] != 'VEH_AGE_MIN'] + [t('VEH_MIN_GE5_P', 'flag')]
# density = 2010 Census population / 2018 Gazetteer land area (km^2): the exact source (ZIPs were selected at >= 1000 per km^2);
# education majority and newest vehicle >= 5 years as probabilities over missing driver/vehicle values; package missing -> mean

# ---------- 19_nonlinear ----------
SPEC9 = SPEC8 + [t('DEN_KM2_SQ', 'linear', 'per-era')]   # density^2 per era (Box-Cox best power 1.25; screen p = 0.04 for the term, 0.10 family-wide)

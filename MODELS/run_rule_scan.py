"""Background job for 16_rule_scan: null calibration + forward rule selection. Results -> models/rule_scan_results.pkl.

Usage (from models/):  ../.venv/bin/python run_rule_scan.py N_NULL MAX_RULES START OUT [THRESHOLD_FROM]
    START: name of a spec in formula_steps (e.g. SPEC4, SPEC6); THRESHOLD_FROM: reuse the threshold of an earlier results file
    (then N_NULL = 0). Importing this module does nothing.
1. Null calibration: simulate y* ~ Bernoulli(p) from SPEC4 fitted on all labeled quotes, refit SPEC4 on y*, run the full scan,
   record the max chi2 per family. Threshold = 95th percentile of the global max: the best score that pure chance produces.
2. Forward selection on the real labels: scan, add the best rule if its chi2 beats the threshold, refit, repeat.
"""
import pickle
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
import formula as FM
import formula_steps as FS
import rule_scan as RS

PAIR_BASE = ['CREDIT_SCORE', 'QUOTED_AMT', 'ACS_POP_DENSITY', 'DRV_AGE_MEAN', 'DRV_HIGH_EDU_SHARE', 'NUM_DRIVERS', 'DRIVERS_PER_VEH',
             'CAT_ZONE', 'VEH_LUXURY_SHARE', 'PKG_ORDINAL', 'DISCOUNT_IND', 'HOME_POLICY_IND', 'VEH_N_ROWS', 'TOTAL_NUM_VEH',
             'NUM_LOANED_VEH', 'NUM_LEASED_VEH', 'NUM_OWNED_VEH', 'PRIOR_CLAIMS_CNT', 'MULTI_QUOTE_IND', 'DAYS_TO_FOLLOWUP',
             'DRV_AGE_MIN', 'DRV_AGE_MAX', 'DRV_YOUNG_SHARE', 'DRV_SAFETY_MEAN', 'DRV_SAFETY_MIN', 'DRV_ACC_FREE_MEAN',
             'DRV_ACC_FREE_MIN', 'DRV_COMMUTE_SUM', 'DRV_COMMUTE_MAX', 'DRV_OWN_SHARE', 'DRV_RENT_SHARE', 'DRV_MALE_SHARE',
             'DRV_MARRIED_SHARE', 'DRV_TELEMATICS_SHARE', 'VEH_AGE_MEAN', 'VEH_MILEAGE_MAX', 'VEH_SAFETY_MEAN', 'VEH_GARAGE_SHARE',
             'VEH_ANTI_THEFT_SHARE', 'ACS_MED_HH_INCOME', 'ACS_OWNER_OCC_SHARE', 'ACS_COMMUTE_MEAN_MIN', 'ACS_MED_AGE', 'QUOTE_MONTH',
             'PRIOR_CARRIER_GRP', 'STATE_CD', 'QUOTE_CHANNEL', 'PRIMARY_PARKING', 'COV_PKG_TYPE']


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def full_scan(spec, L, y, num, cat):
    m = FM.Formula(spec, 0.1, '2era').fit(L, y)
    ctx = RS.Context(m.sc.transform(m._design(L)), m.predict(L), y)
    return RS.scan(ctx, num, cat, L, pair_base=PAIR_BASE), m

def main():
    global N_NULL, MAX_RULES, START, START_NAME, OUT, THR_FROM
    N_NULL, MAX_RULES, START_NAME, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    THR_FROM = sys.argv[5] if len(sys.argv) > 5 else None
    START = getattr(FS, START_NAME)
    L, y = FS.L.copy(), FS.y
    num, cat = RS.build_columns(L)
    log(f'{len(num)} numeric and {len(cat)} categorical candidate columns')
    res = {'pair_base': PAIR_BASE, 'null': [], 'steps': [], 'rules': []}

    # ---- 1. null calibration
    m0 = FM.Formula(START, 0.1, '2era').fit(L, y)
    p0 = m0.predict(L)
    rng = np.random.default_rng(16)
    for i in range(N_NULL):
        ys = (rng.random(len(y)) < p0).astype(float)
        t0 = time.time()
        (df, mx), _ = full_scan(START, L.assign(CONVERT_IND=ys), ys, num, cat)
        res['null'].append({k: float(v) for k, v in mx.items()})
        log(f'null {i + 1}/{N_NULL}: ' + ', '.join(f'{k} {v:.1f}' for k, v in mx.items()) + f' ({time.time() - t0:.0f}s)')
        pickle.dump(res, open(OUT, 'wb'))
    if THR_FROM:
        prev = pickle.load(open(THR_FROM, 'rb'))
        res['null'], thr = prev['null'], prev['threshold']
    G = np.array([n['global'] for n in res['null']])
    thr = float(np.quantile(G, 0.95))
    res['threshold'] = thr
    log(f'threshold (95th pct of null global max): {thr:.1f}; null global max mean {G.mean():.1f}, max {G.max():.1f}')

    # ---- 2. forward selection on real labels
    spec = list(START)
    for it in range(MAX_RULES):
        t0 = time.time()
        (df, mx), m = full_scan(spec, L, y, num, cat)
        top = df.iloc[0]
        res['steps'].append({'iteration': it, 'max': {k: float(v) for k, v in mx.items()}, 'top': df.head(40)[['chi2', 'direction', 'family', 'label', 'rule']].copy()})
        log(f'iteration {it}: best {top.label} chi2 {top.chi2:.1f} ({top.family}) | threshold {thr:.1f} ({time.time() - t0:.0f}s)')
        if top.chi2 <= thr:
            log('no candidate beats the threshold: stop')
            break
        col = f'RULE_{START_NAME}_{it}'
        L[col] = RS.materialize(top.rule, num, cat, L)
        spec = spec + [{'col': col, 'form': 'flag', 'scope': 'all'}]
        res['rules'].append({'col': col, 'rule': top.rule, 'label': top.label, 'chi2': float(top.chi2)})
        res['spec'] = spec
        pickle.dump(res, open(OUT, 'wb'))
    res['spec'] = spec
    pickle.dump(res, open(OUT, 'wb'))
    log(f'done: {len(res["rules"])} rules added')


if __name__ == '__main__':
    main()

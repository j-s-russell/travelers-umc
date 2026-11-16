"""Feature engineering (plan Step 2): target-free features + in-fold target encoding.

- `add_static(df)` adds blocks B-E, which don't use the target, plus string keys for block A. Apply it to labeled and holdout alike.
- `te_fold_fn(keys, make_model)` returns a `fold_fn` for `common.cv_score`: smoothed target encoding learned on the
  training part only. Training rows are themselves encoded out-of-fold (inner 5-fold), so a row never sees its own
  target. Without this the GBM would over-trust the encoding.
  The same function encodes holdout at the end: `fn(labeled, holdout)`.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from common import ROOT, TARGET

# Feature blocks (names of the columns each block adds)
BLOCKS = {
    'B_household': ['DRV_AGE_35_50', 'DRV_AGE_SPREAD', 'N_YOUNG_DRIVERS', 'DRIVERS_PER_VEH'],
    'C_price':     ['QUOTE_PER_DRIVER', 'QUOTE_PER_VEH', 'PRICE_PER_CREDIT'],
    'D_geo':       ['ZIP_LAT', 'ZIP_LON'],
    'E_bundle':    ['BUNDLE_STATUS'],
}
# Block A: high-cardinality keys. LightGBM gets target encodings (TE_<key>); CatBoost gets the raw keys as categoricals.
TE_KEYS = {'A1_zip': ['ZIP_CD', 'ZIP3'], 'A2_combos': ['CARRIER_STATE', 'PKG_CAT']}
MM_COLS = ['TE_MAKE_MODEL_MEAN', 'TE_MAKE_MODEL_MAX']  # block A3: vehicle make/model, encoded per vehicle then rolled up


def add_static(df):
    d = df.copy()
    # B. household structure: average driver age works as a band (35-50 converts ~2x), not a slope
    d['DRV_AGE_35_50'] = d.DRV_AGE_MEAN.between(35, 50).astype('int8')
    d['DRV_AGE_SPREAD'] = d.DRV_AGE_MAX - d.DRV_AGE_MIN          # large spread = parents + teen driver
    d['N_YOUNG_DRIVERS'] = (d.DRV_YOUNG_SHARE * d.NUM_DRIVERS).round()
    d['DRIVERS_PER_VEH'] = d.NUM_DRIVERS / d.VEH_N_ROWS
    # C. price relative to policy size and to credit (low-credit customers were the price-sensitive ones)
    d['QUOTE_PER_DRIVER'] = d.QUOTED_AMT / d.NUM_DRIVERS
    d['QUOTE_PER_VEH'] = d.QUOTED_AMT / d.VEH_N_ROWS
    d['PRICE_PER_CREDIT'] = d.QUOTED_AMT / d.CREDIT_SCORE
    # D. ZIP centroid, so trees can learn regional patterns beyond state and density
    gaz = pd.read_parquet(ROOT / 'data/external/gazetteer2018_zcta.parquet', columns=['INTPTLAT', 'INTPTLONG'])
    d['ZIP_LAT'] = d.ZIP_CD.map(gaz.INTPTLAT).astype(float)
    d['ZIP_LON'] = d.ZIP_CD.map(gaz.INTPTLONG).astype(float)
    # E. bundle status: 0 = no discount, 1 = discount only, 2 = home policy (always comes with a discount)
    d['BUNDLE_STATUS'] = (d.DISCOUNT_IND + d.HOME_POLICY_IND).astype('int8')
    # A. string keys (missing ZIP becomes its own level)
    d['ZIP_CD'] = d.ZIP_CD.fillna('Missing').astype(str)  # pandas keeps NaN through astype(str); CatBoost needs a string
    d['ZIP3'] = d.ZIP3.fillna('Missing').astype(str)
    d['CARRIER_STATE'] = d.PRIOR_CARRIER_GRP.astype(str) + '|' + d.STATE_CD.astype(str)
    d['PKG_CAT'] = d.COV_PKG_TYPE.astype(str) + '|' + d.CAT_ZONE.astype(str)
    return d


def _encode(keys_fit, y_fit, keys_apply, prior, m):
    """Smoothed mean: (sum + m*prior) / (count + m). Unseen levels get the prior."""
    s = y_fit.groupby(keys_fit.to_numpy()).agg(['sum', 'count'])
    enc = (s['sum'] + m * prior) / (s['count'] + m)
    return keys_apply.map(enc).fillna(prior).to_numpy(dtype=float)


_VEH = None
def _vehicles():
    global _VEH
    if _VEH is None:
        _VEH = pd.read_parquet(ROOT / 'data/processed/vehicles.parquet', columns=['POLICY_ID', 'MAKE_MODEL']).astype({'MAKE_MODEL': str})
    return _VEH


def _make_model(fit_pol, apply_ids, m):
    """Encode MAKE_MODEL on the vehicles of `fit_pol` policies, then roll up to mean/max per policy in `apply_ids`."""
    v = _vehicles()
    fit = v[v.POLICY_ID.isin(fit_pol.POLICY_ID)]
    y = fit.POLICY_ID.map(fit_pol.set_index('POLICY_ID')[TARGET])
    app = v[v.POLICY_ID.isin(apply_ids)]
    te = pd.Series(_encode(fit.MAKE_MODEL, y, app.MAKE_MODEL, y.mean(), m), index=app.index)
    g = te.groupby(app.POLICY_ID).agg(['mean', 'max'])
    return g.reindex(apply_ids).to_numpy()


def te_fold_fn(keys=(), make_model=False, m=20, inner=5, seed=0):
    keys = list(keys)

    def fn(tr, sc):
        tr, sc = tr.copy(), sc.copy()
        prior = tr[TARGET].mean()
        for k in keys:
            sc[f'TE_{k}'] = _encode(tr[k], tr[TARGET], sc[k], prior, m)
        if make_model:
            sc[MM_COLS] = _make_model(tr, sc.POLICY_ID, m)
        new = [f'TE_{k}' for k in keys] + (MM_COLS if make_model else [])
        for c in new:
            tr[c] = np.nan
        for a, b in KFold(inner, shuffle=True, random_state=seed).split(tr):
            fit, app = tr.iloc[a], tr.iloc[b]
            for k in keys:
                tr.iloc[b, tr.columns.get_loc(f'TE_{k}')] = _encode(fit[k], fit[TARGET], app[k], prior, m)
            if make_model:
                tr.iloc[b, [tr.columns.get_loc(c) for c in MM_COLS]] = _make_model(fit, app.POLICY_ID, m)
        return tr, sc

    fn.tag = f'te{sorted(keys)}_mm{make_model}_m{m}_in{inner}_s{seed}'  # cache key for common.cv_score
    return fn


# ---------- v2 candidate library (06_features_v2): target-free, built from the raw driver/vehicle tables ----------
def _drivers_v2():
    d = pd.read_parquet(ROOT / 'data/processed/drivers.parquet')
    a = d.DRIVER_AGE
    g = d.assign(
        n_f=d.GENDER.astype(str).eq('F'), n_married=d.MARITAL_STATUS.astype(str).eq('married'),
        n_single=d.MARITAL_STATUS.astype(str).eq('single'), n_divorced=d.MARITAL_STATUS.astype(str).eq('divorced'),
        n_widowed=d.MARITAL_STATUS.astype(str).eq('widowed'), n_own=d.LIVING_STATUS.astype(str).eq('own'),
        n_rent=d.LIVING_STATUS.astype(str).eq('rent'), n_dependent=d.LIVING_STATUS.astype(str).eq('dependent'),
        n_living_missing=d.LIVING_STATUS.astype(str).eq('Missing'),
        n_age_u25=a < 25, n_age_25_35=a.between(25, 34.99), n_age_35_50=a.between(35, 50), n_age_50_65=a.between(50.01, 65),
        n_age_65p=a > 65, n_highedu=d.HIGH_EDU_IND.fillna(0), n_telem=d.TELEMATICS_OPT_IN.fillna(0),
        n_acc_free_0=d.ACCIDENT_FREE_YRS.eq(0)).groupby('POLICY_ID')
    out = g[[c for c in g.obj.columns if c.startswith('n_')]].sum().astype(float).add_prefix('V2_DRV_')
    out['V2_DRV_AGE_STD'] = g.DRIVER_AGE.std()
    out['V2_DRV_AGE_MEDIAN'] = g.DRIVER_AGE.median()
    out['V2_DRV_SAFETY_MAX'] = g.SAFETY_RATING.max()
    out['V2_DRV_SAFETY_STD'] = g.SAFETY_RATING.std()
    out['V2_DRV_ACC_FREE_MAX'] = g.ACCIDENT_FREE_YRS.max()
    out['V2_DRV_ACC_FREE_SUM'] = g.ACCIDENT_FREE_YRS.sum()
    out['V2_DRV_COMMUTE_MEAN'] = g.COMMUTE_MILES_DAILY.mean()
    out['V2_DRV_COMMUTE_MIN'] = g.COMMUTE_MILES_DAILY.min()
    return out


def _vehicles_v2():
    v = pd.read_parquet(ROOT / 'data/processed/vehicles.parquet')
    own, col = v.OWNERSHIP_TYPE.astype(str), v.COLOR.astype(str)
    g = v.assign(n_owned=own.eq('owned'), n_loaned=own.eq('loaned'), n_leased=own.eq('leased'),
                 **{f'n_color_{c}': col.eq(c) for c in col.value_counts().index[:8]},
                 n_new=v.VEHICLE_AGE <= 2, n_old=v.VEHICLE_AGE >= 10, n_antitheft=v.ANTI_THEFT_IND.fillna(0),
                 n_garage=v.GARAGE_IND.fillna(0), n_highmiles=v.ANNUAL_MILEAGE >= 15000).groupby('POLICY_ID')
    out = g[[c for c in g.obj.columns if c.startswith('n_')]].sum().astype(float).add_prefix('V2_VEH_')
    out['V2_VEH_AGE_STD'] = g.VEHICLE_AGE.std()
    out['V2_VEH_MILEAGE_MEAN'] = g.ANNUAL_MILEAGE.mean()
    out['V2_VEH_MILEAGE_MIN'] = g.ANNUAL_MILEAGE.min()
    out['V2_VEH_SAFETY_MAX'] = g.SAFETY_FEATURE_SCORE.max()
    out['V2_VEH_N_MODELS'] = g.MAKE_MODEL.nunique()
    out['V2_VEH_CAR_NO_MAX'] = g.CAR_NO.max()
    return out


def add_v2(df, all_rows=None):
    """Candidate features for 06_features_v2. `all_rows` = labeled + holdout policies, used only for target-free
    frequency counts and the price benchmark (fitted on features, never on the target)."""
    d = df.join(_drivers_v2(), on='POLICY_ID').join(_vehicles_v2(), on='POLICY_ID')
    nd, nv = d.NUM_DRIVERS, d.VEH_N_ROWS
    # cross-table structure (the drivers > vehicles rule was the biggest win so far)
    d['V2_DRV_MINUS_VEH'] = nd - nv
    d['V2_DRV_GT_VEH'] = (nd > nv).astype(int)
    d['V2_ADULTS_GT_VEH'] = ((nd - d.V2_DRV_n_age_u25) > nv).astype(int)
    d['V2_YOUNG_PER_VEH'] = d.V2_DRV_n_age_u25 / nv
    d['V2_VEH_ROWS_MINUS_TOTAL'] = nv - d.TOTAL_NUM_VEH
    d['V2_LOANED_MATCH'] = (d.V2_VEH_n_loaned == d.NUM_LOANED_VEH).astype(int)
    d['V2_LEASED_MATCH'] = (d.V2_VEH_n_leased == d.NUM_LEASED_VEH).astype(int)
    d['V2_MILES_PER_DRIVER'] = d.VEH_MILEAGE_SUM / nd
    d['V2_COMMUTE_PER_VEH'] = d.DRV_COMMUTE_SUM / nv
    # call-note objection triggers (policy-table conditions behind each "why not bound" reason)
    obj = {'LIEN': d.NUM_LOANED_VEH > 0, 'LEASE': d.NUM_LEASED_VEH > 0, 'DRIVERS3': nd >= 3, 'CAT4': d.CAT_ZONE >= 4,
           'LOWPKG': d.COV_PKG_TYPE.astype(str).eq('Low'), 'HIGHPKG': d.COV_PKG_TYPE.astype(str).eq('High'),
           'NODISC': d.DISCOUNT_IND.eq(0), 'NOHOME': d.HOME_POLICY_IND.eq(0)}
    d["V2_N_OBJECTIONS"] = sum(v.fillna(False).astype(int) for v in obj.values())
    # price: affordability and a target-free price benchmark
    d['V2_LOG_PRICE'] = np.log(d.QUOTED_AMT)
    d['V2_PRICE_TO_INCOME'] = d.QUOTED_AMT / d.ACS_MED_HH_INCOME
    d['V2_PRICE_PER_VEH_ROW'] = d.QUOTED_AMT / nv
    src = df if all_rows is None else all_rows
    grp = ['STATE_CD', 'COV_PKG_TYPE']
    med = src.groupby(grp, observed=True).QUOTED_AMT.median().rename('_med')
    d = d.join(med, on=grp)
    d['V2_PRICE_VS_STATE_PKG'] = d.QUOTED_AMT / d._med
    d = d.drop(columns='_med')
    d['V2_PRICE_RANK_STATE'] = d.QUOTED_AMT.rank(pct=True) if all_rows is None else d.join(
        src.groupby('STATE_CD', observed=True).QUOTED_AMT.apply(lambda s: np.sort(s.dropna().to_numpy())).rename('_s'),
        on='STATE_CD').apply(lambda r: np.searchsorted(r._s, r.QUOTED_AMT) / len(r._s) if pd.notna(r.QUOTED_AMT) else np.nan, axis=1)
    # frequency counts (target-free)
    for k in ('ZIP_CD', 'ZIP3'):
        d[f'V2_FREQ_{k}'] = d[k].map(src[k].value_counts()).astype(float)
    # date
    dt = pd.to_datetime(d.QUOTE_DT)
    d['V2_DAY_OF_YEAR'] = dt.dt.dayofyear
    d['V2_DAY_OF_MONTH'] = dt.dt.day
    d['V2_DAYS_SINCE_START'] = (dt - pd.Timestamp('2015-01-01')).dt.days
    return d


def v2_columns(d):
    return [c for c in d.columns if c.startswith('V2_')]


# ---------- categorical versions for CatBoost (07_catboost_cats) ----------
CAT_COPIES = ['NUM_DRIVERS', 'VEH_N_ROWS', 'CAT_ZONE', 'QUOTE_YEAR']
PROFILES = ['C_LIVING_PROFILE', 'C_MARITAL_PROFILE', 'C_AGE_PROFILE', 'C_GENDER_PROFILE']


def add_cats(df):
    """String categoricals so CatBoost can use target statistics and build its own feature combinations:
    - C_<col>: copies of small-integer columns (the numeric originals stay);
    - C_DRV_VEH: '<drivers>_<vehicles>' (the cell of the drivers x vehicles table);
    - household profiles: each driver's living status / marital status / age band / gender, sorted and joined,
      e.g. 'dependent|own|own'. Target-free; apply to labeled and holdout alike."""
    d = df.copy()
    for c in CAT_COPIES:
        d[f'C_{c}'] = d[c].astype('Int64').astype(str)
    d['C_DRV_VEH'] = d.NUM_DRIVERS.astype(str) + '_' + d.VEH_N_ROWS.astype(str)
    dr = pd.read_parquet(ROOT / 'data/processed/drivers.parquet')
    dr['AGE_BAND'] = pd.cut(dr.DRIVER_AGE, [0, 24, 34, 49, 64, 200], labels=['<25', '25-34', '35-49', '50-64', '65+']).astype(str)
    prof = lambda col: dr.groupby('POLICY_ID')[col].agg(lambda s: '|'.join(sorted(s.astype(str))))
    for name, col in zip(PROFILES, ['LIVING_STATUS', 'MARITAL_STATUS', 'AGE_BAND', 'GENDER']):
        d[name] = d.POLICY_ID.map(prof(col)).fillna('none')
    return d


# ---------- more categorical copies (08_cats_extend_tune) ----------
CATS2 = {
    'young drivers':      ['C_N_YOUNG_DRIVERS'],
    'policy veh counts':  ['C_TOTAL_NUM_VEH', 'C_NUM_OWNED_VEH', 'C_NUM_LOANED_VEH', 'C_NUM_LEASED_VEH'],
    'prior claims':       ['C_PRIOR_CLAIMS_CNT'],
    'quote month':        ['C_QUOTE_MONTH'],
    'drivers - vehicles': ['C_DRV_MINUS_VEH'],
    'driver status counts': ['C_N_DEPENDENT', 'C_N_MARRIED', 'C_N_OWN', 'C_N_RENT'],
    'binned credit/age/price': ['C_CREDIT_BAND', 'C_AGE_MEAN_BAND', 'C_PRICE_BAND'],
}


def add_cats2(df):
    """More string copies of small-integer columns (and coarse bands of the three strongest continuous ones),
    on top of add_cats. Target-free; fixed bin edges, so labeled and holdout get identical levels."""
    d = add_cats(df)
    s = lambda x: x.astype('Int64').astype(str)
    d['C_N_YOUNG_DRIVERS'] = s(d.N_YOUNG_DRIVERS)
    for c in ['TOTAL_NUM_VEH', 'NUM_OWNED_VEH', 'NUM_LOANED_VEH', 'NUM_LEASED_VEH', 'PRIOR_CLAIMS_CNT', 'QUOTE_MONTH']:
        d[f'C_{c}'] = s(d[c])
    d['C_DRV_MINUS_VEH'] = s((d.NUM_DRIVERS - d.VEH_N_ROWS).clip(-5, 4))
    dr = pd.read_parquet(ROOT / 'data/processed/drivers.parquet')
    cnt = lambda col, v: dr.assign(x=dr[col].astype(str).eq(v)).groupby('POLICY_ID').x.sum()
    for name, col, v in [('C_N_DEPENDENT', 'LIVING_STATUS', 'dependent'), ('C_N_MARRIED', 'MARITAL_STATUS', 'married'),
                         ('C_N_OWN', 'LIVING_STATUS', 'own'), ('C_N_RENT', 'LIVING_STATUS', 'rent')]:
        d[name] = s(d.POLICY_ID.map(cnt(col, v)).fillna(0))
    d['C_CREDIT_BAND'] = pd.cut(d.CREDIT_SCORE, [0, 500, 550, 600, 650, 700, 750, 2000]).astype(str)
    d['C_AGE_MEAN_BAND'] = pd.cut(d.DRV_AGE_MEAN, [0, 25, 30, 35, 40, 45, 50, 55, 60, 70, 200]).astype(str)
    d['C_PRICE_BAND'] = pd.cut(d.QUOTED_AMT, [0, 1000, 2000, 3000, 4000, 6000, 9000, 15000, 1e9]).astype(str)
    return d


# ---------- generator eras (10_generator) ----------
def add_era(df):
    """Conversion logic changes on 2017-01-01 (price/credit effects vanish, education/urban effects appear) and the discount
    effect exists only in 2016, so the generator behaves like three eras: 2015, 2016, 2017-18."""
    d = df.copy()
    d['ERA'] = np.select([d.QUOTE_YEAR <= 2015, d.QUOTE_YEAR == 2016], ['2015', '2016'], '2017-18')
    return d


# ---------- recovered generator rules (11_rules) ----------
RULES = {
    'edu majority x era':  ['R_EDU_MAJ', 'R_EDU_MAJ_LATE'],
    '2016 discount/home':  ['R_DISC_2016', 'R_HOME_2016'],
    'drivers per vehicle': ['R_DPV_GT1', 'R_DPV_HALF_TO_1'],
    'density x era':       ['R_LOG_DENSITY_LATE', 'R_DENSE_LATE'],
    'price/credit x era':  ['R_LOG_PRICE_EARLY', 'R_CREDIT_EARLY'],
    'drivers x era':       ['R_3_DRIVERS_LATE'],
}


def add_rules(df):
    """Flags for the rules recovered in 10_generator / 11_rules (needs add_static + add_era columns). Target-free.
    - education: in 2017-18 conversion jumps when MORE THAN HALF of the drivers have higher education (exactly half doesn't count);
    - discount and home policy matter only in 2016;
    - drivers per vehicle: > 1 converts ~5%, (0.5, 1] ~14%, < 0.5 ~10.5%;
    - density: strong, stepped rise in 2017-18 only; price and credit matter mostly before 2017;
    - 3-driver households convert best in 2017-18 (worse than 1-2 drivers before)."""
    d = df.copy()
    late, y16 = d.ERA.eq('2017-18'), d.ERA.eq('2016')
    d['R_EDU_MAJ'] = (d.DRV_HIGH_EDU_SHARE > 0.5).astype(int)
    d['R_EDU_MAJ_LATE'] = (d.R_EDU_MAJ.eq(1) & late).astype(int)
    d['R_DISC_2016'] = (d.DISCOUNT_IND.eq(1) & y16).astype(int)
    d['R_HOME_2016'] = (d.HOME_POLICY_IND.eq(1) & y16).astype(int)
    d['R_DPV_GT1'] = (d.DRIVERS_PER_VEH > 1).astype(int)
    d['R_DPV_HALF_TO_1'] = d.DRIVERS_PER_VEH.between(0.5, 1).astype(int)
    d['R_LOG_DENSITY_LATE'] = np.log1p(d.ACS_POP_DENSITY).where(late, 0.0)
    d['R_DENSE_LATE'] = (pd.cut(d.ACS_POP_DENSITY, [-1, 10000, 20000, 30000, 40000, np.inf], labels=False).fillna(0) * late).astype(int)
    d['R_LOG_PRICE_EARLY'] = np.log(d.QUOTED_AMT).where(~late, np.nan)
    d['R_CREDIT_EARLY'] = d.CREDIT_SCORE.where(~late, np.nan)
    d['R_3_DRIVERS_LATE'] = (d.NUM_DRIVERS.eq(3) & late).astype(int)
    return d

"""Explicit logistic "formula" models built from a readable spec (12_formula).

A spec is a list of terms (JSON-serializable dicts, so `common.cv_score` can cache on them):
    {'col': 'CREDIT_SCORE', 'form': 'linear', 'scope': 'early'}
- form:  'linear' | 'flag' (0/1 as is) | 'spline' (k knots) | 'step' (cuts, left-closed bins) | 'hinge' (knots) | 'onehot'
- tf:    optional transform before the form: 'log' | 'log1p'
- scope: 'all' | 'early' (2015+2016) | 'late' (2017-18) | '2015' | '2016' | 'split' (separate early and late copies)
         | 'per-era' (separate copy for each of the 3 eras) | 'per-year' (2015, 2016, 2017, 2018) | 'Y2017' / 'Y2018' (one year)
- na:    True adds a missing-value indicator (scope 'all'); missing values are otherwise imputed with the training median.
Era intercepts are always included: `intercepts` = '3era' (default: 2015 / 2016 / 2017-18), '2era' (early / late) or '4era' (years). Columns are standardized, then an L2 logistic regression with strength `C` is fitted.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler

import features as FE

ERAS = ['2015', '2016', '2017-18']
SCOPES = {'all': [None], 'early': ['early'], 'late': ['2017-18'], '2015': ['2015'], '2016': ['2016'],
          'split': ['early', '2017-18'], 'per-era': ERAS, 'per-year': ['Y2015', 'Y2016', 'Y2017', 'Y2018'],
          'Y2017': ['Y2017'], 'Y2018': ['Y2018']}


def prep(df):
    """All columns a formula may use: static features, ERA, recovered rule flags, plus a few derived columns."""
    d = FE.add_rules(FE.add_era(FE.add_static(df)))
    d['NUM_DRIVERS_C5'] = d.NUM_DRIVERS.clip(upper=5)
    d['N_EDU_DRIVERS'] = (d.DRV_HIGH_EDU_SHARE * d.NUM_DRIVERS).round()
    d['PKG_ORDINAL'] = d.COV_PKG_TYPE.astype(str).map({'Low': -1, 'Medium': 0, 'High': 1})      # Missing -> NaN -> imputed
    # rule found by the exhaustive scan (16_rule_scan): in 2016, prior carriers 3 and 8 and 'no prior carrier' convert less
    d['R_CARRIER_3_8_MISS_2016'] = (d.PRIOR_CARRIER_GRP.astype(str).isin(['Carrier_3', 'Carrier_8', 'Missing']) & d.ERA.eq('2016')).astype(int)
    # 17_external_sources: generator density (2010 Census / 2018 land, per km^2) and probability-imputed flags
    import impute
    d = impute.core(d)
    d['DEN_KM2_SQ'] = d.DEN_KM2 ** 2          # 19_nonlinear: density effect is slightly convex (SPEC9)
    return d


def columns(spec):
    return sorted({t['col'] for t in spec} | {'ERA', 'QUOTE_YEAR'})


def _mask(d, era):
    if era is None:
        return np.ones(len(d))
    if isinstance(era, str) and era.startswith('Y'):
        return d.QUOTE_YEAR.eq(int(era[1:])).to_numpy(float)
    if era == 'early':
        return d.ERA.isin(['2015', '2016']).to_numpy(float)
    return d.ERA.eq(era).to_numpy(float)


def _label(t):
    s = t['col'] + (f"[{t['tf']}]" if t.get('tf') else '')
    return f"{s}:{t['form']}" + (f"{t.get('cuts') or t.get('knots') or t.get('k') or ''}" if t['form'] in ('step', 'hinge', 'spline') else '')


class Formula:
    def __init__(self, spec, C=1.0, intercepts='3era'):
        self.spec, self.C, self.intercepts = spec, C, intercepts

    # --- one term's base columns (before scope) ---
    def _base(self, i, t, d, fit):
        st = self.state.setdefault(i, {}) if fit else self.state[i]
        x = d[t['col']]
        if t['form'] == 'onehot':
            x = x.astype(object).where(x.notna(), 'Missing').astype(str)
            if fit:
                vc = x.value_counts()
                st['levels'] = [v for v in vc.index if v != vc.index[0]]          # most frequent level = reference
            return np.column_stack([x.eq(v).to_numpy(float) for v in st['levels']]), [f'={v}' for v in st['levels']]
        x = x.astype(float)
        if t.get('tf') == 'log':
            x = np.log(x.clip(lower=1e-9))
        elif t.get('tf') == 'log1p':
            x = np.log1p(x)
        if fit:
            st['med'] = float(x.median())
        miss = x.isna().to_numpy(float)
        x = x.fillna(st['med']).to_numpy()
        f = t['form']
        if f in ('linear', 'flag'):
            cols, names = x[:, None], ['']
        elif f == 'spline':
            if fit:
                st['spl'] = SplineTransformer(n_knots=t.get('k', 5), degree=3, knots='quantile', extrapolation='constant').fit(x[:, None])
            cols = st['spl'].transform(x[:, None]); names = [f'_b{j}' for j in range(cols.shape[1])]
        elif f == 'step':
            edges = [-np.inf] + list(t['cuts']) + [np.inf]
            b = np.digitize(x, edges[1:-1], right=False)                       # bin index, left-closed bins
            cols = np.column_stack([(b == j).astype(float) for j in range(1, len(edges) - 1)])
            names = [f'>={c}' for c in t['cuts']]
        elif f == 'hinge':
            cols = np.column_stack([x] + [np.maximum(0, x - k) for k in t['knots']])
            names = [''] + [f'_over{k}' for k in t['knots']]
        else:
            raise ValueError(f)
        return cols, names, miss

    def _design(self, d, fit=False):
        groups = {'3era': ERAS[1:], '2era': ['2017-18'], '4era': ['Y2016', 'Y2017', 'Y2018']}[self.intercepts]
        blocks, names = [np.column_stack([_mask(d, e) for e in groups])], [f'era={e}' for e in groups]
        for i, t in enumerate(self.spec):
            out = self._base(i, t, d, fit)
            cols, nm = out[0], out[1]
            for era in SCOPES[t.get('scope', 'all')]:
                blocks.append(cols * _mask(d, era)[:, None])
                names += [f"{_label(t)}{n}" + (f' @{era}' if era else '') for n in nm]
            if len(out) == 3 and t.get('na'):
                blocks.append(out[2][:, None]); names.append(f"{t['col']}:missing")
        X = np.hstack(blocks)
        if fit:
            self.names = names
        return X

    def fit(self, d, y):
        self.state = {}
        X = self._design(d, fit=True)
        self.sc = StandardScaler().fit(X)
        sd = self.sc.scale_.copy(); sd[sd == 0] = 1; self.sc.scale_ = sd
        self.m = LogisticRegression(C=self.C, max_iter=10000).fit(self.sc.transform(X), np.asarray(y))
        return self

    def decision(self, d):
        return self.m.decision_function(self.sc.transform(self._design(d)))

    def predict(self, d):
        return self.m.predict_proba(self.sc.transform(self._design(d)))[:, 1]

    def coef_table(self):
        """Coefficients on the original (unstandardized) scale of each design column."""
        b = self.m.coef_[0] / self.sc.scale_
        return pd.DataFrame({'term': self.names, 'coef': b, 'coef (std)': self.m.coef_[0]}).set_index('term')

    @property
    def n_params(self):
        return len(self.names)

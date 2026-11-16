"""Download-and-cache helpers for alternative ZIP-level sources (17_external_sources). Every getter returns a DataFrame indexed by
5-character ZCTA (or by county FIPS for county tables), cached as data/external/<name>.parquet; delete a file to re-download it.
Needs CENSUS_API_KEY in .env for the Census API (the key is never printed)."""
import io, os, re, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / 'data/external'
load_dotenv(ROOT / '.env')
STATES = {'AL': '01', 'CT': '09', 'FL': '12', 'GA': '13', 'MN': '27', 'NJ': '34', 'NY': '36', 'WI': '55'}
GAZ = 'https://www2.census.gov/geo/docs/maps-data/data/gazetteer'


def _cache(name, fn):
    f = EXT / f'{name}.parquet'
    if not f.exists():
        fn().to_parquet(f)
    return pd.read_parquet(f)


def _api(url, var, per_state=True):
    key = os.environ['CENSUS_API_KEY']
    calls = [{'in': f'state:{s}'} for s in STATES.values()] if per_state else [{}]
    frames = []
    for extra in calls:
        r = requests.get(url, params={'get': var, 'for': 'zip code tabulation area:*', 'key': key, **extra}, timeout=180)
        r.raise_for_status()
        j = r.json()
        frames.append(pd.DataFrame(j[1:], columns=j[0]))
    df = pd.concat(frames).rename(columns={'zip code tabulation area': 'ZCTA'}).drop_duplicates('ZCTA').set_index('ZCTA')
    return pd.to_numeric(df[var], errors='coerce').mask(lambda s: s < 0).rename('POP').to_frame()


def acs_pop(year):
    """ACS 5-year total population (B01003) by ZCTA. ZCTAs nest in states up to the 2019 vintage; national query after."""
    return _cache(f'acs{year}_pop', lambda: _api(f'https://api.census.gov/data/{year}/acs/acs5', 'B01003_001E', per_state=year <= 2019))


def dec2020_pop():
    return _cache('dec2020_pop', lambda: _api('https://api.census.gov/data/2020/dec/dhc', 'P1_001N', per_state=False))


def gazetteer(year):
    """Census Gazetteer ZCTA file: ALAND_SQMI, AWATER_SQMI, INTPTLAT, INTPTLONG (2010 also POP10)."""
    url = f'{GAZ}/Gaz_zcta_national.zip' if year == 2010 else f'{GAZ}/{year}_Gazetteer/{year}_Gaz_zcta_national.zip'

    def get():
        z = zipfile.ZipFile(io.BytesIO(requests.get(url, timeout=180).content))
        g = pd.read_csv(z.open(z.namelist()[0]), sep='\t', dtype={'GEOID': str})
        g.columns = g.columns.str.strip()
        return g.rename(columns={'GEOID': 'ZCTA'}).set_index('ZCTA')
    return _cache(f'gazetteer{year}_zcta' if year != 2018 else 'gazetteer2018_zcta', get)


def zcta_county_2010():
    """2010 ZCTA-to-county relationship file: one row per ZCTA x county piece with populations and land areas."""
    def get():
        r = requests.get('https://www2.census.gov/geo/docs/maps-data/data/rel/zcta_county_rel_10.txt', timeout=180)
        r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text), dtype={'ZCTA5': str, 'STATE': str, 'COUNTY': str, 'GEOID': str})
        return d.rename(columns={'ZCTA5': 'ZCTA'}).reset_index(drop=True)
    return _cache('zcta_county_rel_2010', get)


def fema_declarations():
    """OpenFEMA disaster declarations 2000-2018 for the 8 states: number of distinct declared disasters per county and incident type."""
    def get():
        rows = []
        for st in STATES:
            skip = 0
            while True:
                r = requests.get('https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries', timeout=180, params={
                    '$filter': f"state eq '{st}' and declarationDate ge '2000-01-01T00:00:00.000Z' and declarationDate le '2018-12-31T23:59:59.000Z'",
                    '$select': 'disasterNumber,incidentType,fipsStateCode,fipsCountyCode', '$top': 10000, '$skip': skip})
                r.raise_for_status()
                page = r.json()['DisasterDeclarationsSummaries']
                rows += page
                if len(page) < 10000:
                    break
                skip += 10000
        d = pd.DataFrame(rows)
        d['COUNTY_FIPS'] = d.fipsStateCode + d.fipsCountyCode
        d = d[d.fipsCountyCode != '000'].drop_duplicates(['COUNTY_FIPS', 'disasterNumber'])
        return d.pivot_table(index='COUNTY_FIPS', columns='incidentType', values='disasterNumber', aggfunc='count', fill_value=0)
    return _cache('fema_declarations_2000_2018', get)


def density_table():
    """All density variants by ZCTA (people per sq mi of land unless noted) plus helper columns."""
    g10, g18, g20 = gazetteer(2010), gazetteer(2018), gazetteer(2020)
    land = {'g10': g10.ALAND_SQMI, 'g18': g18.ALAND_SQMI, 'g20': g20.ALAND_SQMI}
    tot18 = g18.ALAND_SQMI + g18.AWATER_SQMI
    pops = {f'acs{y}': acs_pop(y).POP for y in range(2011, 2020)}
    pops |= {'acs2020': acs_pop(2020).POP, 'dec2020': dec2020_pop().POP, 'dec2010': g10.POP10.astype(float)}
    out = {}
    for k, p in pops.items():
        lk = 'g20' if k in ('acs2020', 'dec2020') else ('g10' if k in ('dec2010', 'acs2011', 'acs2012') else 'g18')
        out[f'D_{k}'] = p / land[lk].reindex(p.index)
    out['D_acs2018_landg10'] = pops['acs2018'] / g10.ALAND_SQMI.reindex(pops['acs2018'].index)
    out['D_acs2018_total_area'] = pops['acs2018'] / tot18.reindex(pops['acs2018'].index)
    T = pd.DataFrame(out).replace([np.inf, -np.inf], np.nan)
    # county density (2010 relationship file), assigned to each ZCTA by its largest-population county piece
    rel = zcta_county_2010()
    main = rel.sort_values('ZPOPPCT', ascending=False).drop_duplicates('ZCTA').set_index('ZCTA')
    T['COUNTY_FIPS'] = main.GEOID.reindex(T.index)
    T['D_county2010'] = (main.COPOP / (main.COAREALAND / 2.589988e6)).reindex(T.index)
    T['LAT'], T['LON'] = g18.INTPTLAT.reindex(T.index), g18.INTPTLONG.reindex(T.index)
    T['WATER_SHARE'] = (g18.AWATER_SQMI / tot18).reindex(T.index)
    return T


SF1_VARS = {'P001001': 'pop', 'P002002': 'urban', 'P002003': 'urbanized_area', 'P002004': 'urban_cluster', 'P002005': 'rural',
            'P003002': 'white', 'P003003': 'black', 'P003005': 'asian', 'P004003': 'hispanic', 'P012002': 'male',
            'P013001': 'median_age', 'P017001': 'avg_hh_size', 'P037001': 'avg_family_size', 'P018001': 'households',
            'P018002': 'family_hh', 'P018003': 'married_family_hh', 'P018008': 'nonfamily_hh', 'P020002': 'hh_with_children',
            'P021002': 'hh_with_seniors', 'H001001': 'housing_units', 'H003002': 'occupied', 'H003003': 'vacant',
            'H004002': 'owned_mortgage', 'H004003': 'owned_free', 'H004004': 'renter', 'H005006': 'seasonal_vacant',
            'P042001': 'group_quarters'}


def dec2010_sf1():
    """2010 Census SF1 counts by ZCTA (national query), renamed per SF1_VARS."""
    def get():
        key = os.environ['CENSUS_API_KEY']
        r = requests.get('https://api.census.gov/data/2010/dec/sf1', timeout=300,
                         params={'get': ','.join(SF1_VARS), 'for': 'zip code tabulation area:*', 'key': key})
        r.raise_for_status()
        j = r.json()
        d = pd.DataFrame(j[1:], columns=j[0]).rename(columns={'zip code tabulation area': 'ZCTA'}).set_index('ZCTA')
        return d[list(SF1_VARS)].apply(pd.to_numeric, errors='coerce').rename(columns=SF1_VARS)
    return _cache('dec2010_sf1', get)


def acs_tables(year, tables):
    """ACS 5-year detailed tables (all estimate columns) by ZCTA, queried per state like 02_external_census."""
    out = {}
    for tb in tables:
        def get(tb=tb):
            key = os.environ['CENSUS_API_KEY']
            frames = []
            for fips in STATES.values():
                r = requests.get(f'https://api.census.gov/data/{year}/acs/acs5', timeout=180,
                                 params={'get': f'group({tb})', 'for': 'zip code tabulation area:*', 'in': f'state:{fips}', 'key': key})
                r.raise_for_status()
                j = r.json()
                frames.append(pd.DataFrame(j[1:], columns=j[0]))
            df = pd.concat(frames).rename(columns={'zip code tabulation area': 'ZCTA'}).drop_duplicates('ZCTA').set_index('ZCTA')
            est = [c for c in df if c.startswith(tb) and c.endswith('E')]
            return df[est].apply(pd.to_numeric, errors='coerce').mask(lambda x: x < 0)
        out[tb] = _cache(f'acs{year}_{tb}', get)
    return out


def area_variables():
    """Candidate area-level variables by ZCTA: 2010 SF1 shares/densities and ACS 2011 versions of the 02 features."""
    s = dec2010_sf1()
    land = gazetteer(2018).ALAND / 1e6
    p, h, hh = s['pop'], s.housing_units, s.households
    V = pd.DataFrame({
        'sf1_housing_density': h / land.reindex(s.index), 'sf1_household_density': hh / land.reindex(s.index),
        'sf1_urban_share': s.urban / p, 'sf1_urbanized_share': s.urbanized_area / p, 'sf1_rural_share': s.rural / p,
        'sf1_white_share': s.white / p, 'sf1_black_share': s.black / p, 'sf1_asian_share': s.asian / p, 'sf1_hispanic_share': s.hispanic / p,
        'sf1_male_share': s.male / p, 'sf1_median_age': s.median_age, 'sf1_avg_hh_size': s.avg_hh_size, 'sf1_avg_family_size': s.avg_family_size,
        'sf1_family_share': s.family_hh / hh, 'sf1_married_share': s.married_family_hh / hh, 'sf1_nonfamily_share': s.nonfamily_hh / hh,
        'sf1_children_hh_share': s.hh_with_children / hh, 'sf1_senior_hh_share': s.hh_with_seniors / hh,
        'sf1_vacancy_rate': s.vacant / h, 'sf1_seasonal_share': s.seasonal_vacant / h, 'sf1_owner_share': (s.owned_mortgage + s.owned_free) / s.occupied,
        'sf1_owned_free_share': s.owned_free / s.occupied, 'sf1_renter_share': s.renter / s.occupied, 'sf1_group_quarters_share': s.group_quarters / p,
        'sf1_pop': p, 'sf1_land_km2': land.reindex(s.index)})
    a = acs_tables(2011, ['B19013', 'B01002', 'B25003', 'B08201', 'B08301', 'B08303', 'B15002', 'B25077', 'B25064', 'B17001'])
    cm, tt, ed, vh = a['B08301'], a['B08303'], a['B15002'], a['B08201']
    A = pd.DataFrame({
        'acs11_med_income': a['B19013'].B19013_001E, 'acs11_med_age': a['B01002'].B01002_001E,
        'acs11_owner_share': a['B25003'].B25003_002E / a['B25003'].B25003_001E,
        'acs11_no_vehicle_share': vh.B08201_002E / vh.B08201_001E,
        'acs11_veh_per_hh': vh[[f'B08201_{i:03d}E' for i in range(3, 7)]] @ [1, 2, 3, 4] / vh.B08201_001E,
        'acs11_drove_alone': cm.B08301_003E / cm.B08301_001E, 'acs11_carpool': cm.B08301_004E / cm.B08301_001E,
        'acs11_transit': cm.B08301_010E / cm.B08301_001E, 'acs11_wfh': cm.B08301_021E / cm.B08301_001E,
        'acs11_commute_45plus': tt[[f'B08303_{i:03d}E' for i in range(11, 14)]].sum(axis=1) / tt.B08303_001E,
        'acs11_bachelor_plus': (ed[[f'B15002_{i:03d}E' for i in range(15, 19)]].sum(axis=1) + ed[[f'B15002_{i:03d}E' for i in range(32, 36)]].sum(axis=1)) / ed.B15002_001E,
        'acs11_med_home_value': a['B25077'].B25077_001E, 'acs11_med_rent': a['B25064'].B25064_001E,
        'acs11_poverty_share': a['B17001'].B17001_002E / a['B17001'].B17001_001E})
    return V.join(A, how='left').replace([np.inf, -np.inf], np.nan)

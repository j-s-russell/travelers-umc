"""uszipcode (simple DB, exported to data/external/uszipcode_simple.parquet) feature screen vs SPEC8, all era scopes, pure-chance threshold. Run from models/."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, formula as FM, formula_steps as FS, area_screen as AS
U=pd.read_parquet('../data/external/uszipcode_simple.parquet').set_index('zipcode')
L,y=FS.L,FS.y; z=L.ZIP_CD.astype(str).to_numpy(); J=U.reindex(z)
F={'median_household_income':J.median_household_income,'median_home_value':J.median_home_value,'housing_units':J.housing_units,
   'occupied_housing_units':J.occupied_housing_units,'occupancy_rate':J.occupied_housing_units/J.housing_units,'housing_density':J.housing_units/J.land_area_in_sqmi,
   'water_area':J.water_area_in_sqmi,'water_share':J.water_area_in_sqmi/(J.water_area_in_sqmi+J.land_area_in_sqmi),'land_area':J.land_area_in_sqmi,
   'radius_in_miles':J.radius_in_miles,'lat':J.lat,'lng':J.lng,'population':J.population,'pop_per_housing_unit':J.population/J.housing_units,
   'income_to_home_value':J.median_household_income/J.median_home_value,'type_STANDARD':(J.zipcode_type=='STANDARD').astype(float),
   'type_UNIQUE':(J.zipcode_type=='UNIQUE').astype(float),'type_POBOX':(J.zipcode_type=='PO BOX').astype(float),
   'income_missing':J.median_household_income.isna().astype(float),'density_missing':J.population_density.isna().astype(float)}
F=pd.DataFrame({k:np.asarray(v,dtype=float) for k,v in F.items()},index=L.index).replace([np.inf,-np.inf],np.nan)
print('coverage of policies:', F.notna().mean().round(3).to_dict())
for c in list(F.columns):
    if F[c].min()>=0 and F[c].nunique()>2: F['log '+c]=np.log1p(F[c])
era=L.ERA.astype(str).to_numpy(); masks={'all':np.ones(len(y),bool),'early':era!='2017-18','late':era=='2017-18','2015':era=='2015','2016':era=='2016'}
cols,names=[],[]
X0=F.fillna(F.median()).to_numpy(float)
for sc,mk in masks.items():
    Xs=(X0-X0[mk].mean(0))/(X0[mk].std(0)+1e-12); Xs[~mk]=0; cols.append(Xs); names+=[f'{c} @{sc}' for c in F.columns]
Xc=np.hstack(cols)
chi,U_=AS.score_chi2(FS.SPEC8,L,y,Xc)
m=FM.Formula(FS.SPEC8,0.1,'2era').fit(L,y); p=m.predict(L); rng=np.random.default_rng(21)
null=[AS.score_chi2(FS.SPEC8,L.assign(CONVERT_IND=ys),ys,Xc)[0].max() for ys in ((rng.random(len(y))<p).astype(float) for _ in range(100))]
print(f'{len(names)} candidates; pure-chance threshold {np.quantile(null,0.95):.1f} (null max mean {np.mean(null):.1f})')
s=pd.Series(chi,index=names); sign=pd.Series(np.sign(U_),index=names)
print(pd.DataFrame({'chi2':s,'direction':sign.map({1.0:'+',-1.0:'-',0.0:''})}).sort_values('chi2',ascending=False).head(15).round(1).to_string())
F.to_parquet('/private/tmp/claude-501/-Users-joerussell-Desktop-PROJECTS-competitions-travelers-umc/4769a44c-c59d-4475-9fff-bcd5ad48a0b9/scratchpad/usz_feats.parquet')

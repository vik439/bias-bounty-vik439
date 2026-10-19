"""Candidate per-tract signals built only from provided layers. Each is in [0,1]; written to out/candidates.parquet."""
import numpy as np, pandas as pd, re, glob
f=pd.read_parquet('data/feat-all2.parquet'); r=pd.read_parquet('data/routes-long.parquet')
HW=['motorway','trunk','primary','secondary']
f['km_hw']=f[HW and ['km_'+c for c in HW]].sum(axis=1)
STATE=re.compile(r'^US:[A-Z]{2}$'); SUB={'FM','RM','Loop','Spur','Toll','Business','Park','PA','Beltway','NASA','Turnpike','Scenic','Alternate','Truck','Bypass'}
def core(n):
    p=n.split(':')
    return n in('US:I','US:US') or bool(STATE.match(n)) or (len(p)>=3 and p[0]=='US' and (p[1] in('I','US') or len(p[1])==2) and p[2] in SUB)
ok={n:any(core(x) for x in n.split('|')) for n in r.net.unique()}
rc=r[r.net.map(ok)]
f['km_T_core']=f.GEOID.map(rc.groupby('GEOID')['km'].sum()).fillna(0)
f['km_core_hw']=f.GEOID.map(rc[rc['class'].isin(HW)].groupby('GEOID')['km'].sum()).fillna(0)
def gap(num,den,thr=0.05):
    d=den>thr; return np.where(d,1-(num/den.replace(0,np.nan)).clip(upper=1),0.0), d
C=pd.DataFrame({'GEOID':f.GEOID,'region':f.region})
C['t_all'],C['t_all_def']=gap(f.km_hw,f.km_T_all)
C['t_core'],C['t_core_def']=gap(f.km_hw,f.km_T_core)
C['t_core_strict'],_=gap(f.km_core_hw,f.km_T_core)          # route roads that are themselves below highway class
ref=f.usfs_BuildingCount_sum.fillna(0); C['b_usfs'],C['b_def']=gap(f.bld_n,ref,0)
C['b_ms_share']=((f.bld_ms)/f.bld_n.replace(0,np.nan)).fillna(0)   # machine-derived share, diagnostic
# facility evidence
ev_fire=np.maximum(f.bldcls_fire_station,0)+f.poi_fire_alt; ev_sch=f.bldcls_school+f.poi_school_alt; ev_ems=f.poi_ems_alt
for k,n,ev in [('fire',f.poi_fire,ev_fire),('school',f.poi_school,ev_sch),('ems',f.poi_ems,ev_ems)]:
    C['fac_'+k+'_hidden']=((n==0)&(ev>0)).astype(float); C['no_'+k+'_poi']=(n==0).astype(float)
    C['fac_'+k+'_partial']=np.where((n+ev)>0,1-(n/(n+ev).replace(0,np.nan)).clip(upper=1),0.0)
hu=f.housing_units.fillna(0)
C['cbp_flat'],_=gap(f.poi_n,0.055*hu,0)
f['county']=f.GEOID.str[:5]; cp=f.groupby('county').poi_n.transform('sum'); ch=f.groupby('county').housing_units.transform('sum')
C['cbp_share2'],_=gap(f.poi_n,0.5*cp*hu/ch.replace(0,np.nan),0); C['cbp_share3'],_=gap(f.poi_n,(1/3)*cp*hu/ch.replace(0,np.nan),0)
pc=f.groupby('county').pop_total.transform('sum'); C['cbp_popshare2'],_=gap(f.poi_n,0.5*cp*f.pop_total/pc.replace(0,np.nan),0)
C['rural']=(f.ur_class=='Rural').astype(float); C['tribal']=f.tribal_any.astype(float)
area=f.ALAND/1e6; C['log_area']=(np.log10(area.clip(lower=0.1))/4).clip(0,1); C['log_dens']=(np.log10((f.pop_total/area.replace(0,np.nan)).fillna(0).clip(lower=1))/4).clip(0,1)
C['n_defined_2']=(~C.t_all_def).astype(float)          # roads undefined -> divisor 2
C=C.fillna(0); C.to_parquet('out/candidates.parquet')
num=C.select_dtypes('number'); print(num.describe().T[['mean','std']].round(3).to_string())
print(num.corr().round(2).loc[['t_all','fac_fire_hidden','no_fire_poi','cbp_flat','cbp_share2','rural','log_area'],['t_all','t_core','b_usfs','fac_fire_hidden','no_fire_poi','cbp_flat','cbp_share2','rural','log_area','n_defined_2']].to_string())

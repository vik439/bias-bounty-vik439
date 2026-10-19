"""Rule-based coverage-gap estimate from provided layers only. No labels exist; deterministic."""
import numpy as np, pandas as pd
f=pd.read_parquet('data/feat-all.parquet')
f['km_hw']=f[['km_motorway','km_trunk','km_primary','km_secondary']].sum(axis=1)
# building: Overture count vs USFS building count (provided strata; Microsoft-derived)
ref=f.usfs_BuildingCount_sum.fillna(0)
f['b_def']=ref>0; f['bg']=np.where(f.b_def,1-(f.bld_n/ref.replace(0,np.nan)).clip(upper=1),0.0)
# transport: Overture named-highway classes vs route-tagged network as stand-in for the highway reference
T=f.km_T_all; f['t_def']=T>0.05
f['tg']=np.where(f.t_def,1-(f.km_hw/T.replace(0,np.nan)).clip(upper=1),0.0)
# poi: facility halves use in-data evidence that a facility exists but lacks the scored primary category
def fac(n,alt):
    d=(n>0)|(alt>0); return d, np.where(n>0,0.0,1.0)
fd,fg=fac(f.poi_fire,f.poi_fire_alt+f.bldcls_fire_station)
sd,sg=fac(f.poi_school,f.poi_school_alt+f.bldcls_school)
ed,eg=fac(f.poi_ems,f.poi_ems_alt)
G=np.vstack([np.where(fd,fg,np.nan),np.where(sd,sg,np.nan),np.where(ed,eg,np.nan)])
hd=(~np.isnan(G)).any(0); hif=np.where(hd,np.nanmean(np.where(hd,G,0.0),axis=0),np.nan)
est=0.055*f.housing_units.fillna(0)            # establishments stand-in
cd=est>0; cbp=np.where(cd,1-(f.poi_n/est.replace(0,np.nan)).clip(upper=1),np.nan)
P=np.vstack([hif,cbp]); f['p_def']=(~np.isnan(P)).any(0)
f['pg']=np.where(f.p_def,np.nanmean(np.where(f.p_def,P,0.0),axis=0),0.0)
D=f[['t_def','b_def','p_def']].values.astype(float); V=f[['tg','bg','pg']].values
f['score']=np.where(D.sum(1)>0,(V*D).sum(1)/np.maximum(D.sum(1),1),0.0)
f[['GEOID','region','tg','bg','pg','t_def','b_def','p_def','score']].to_parquet('out/components.parquet')
print(f.groupby('region')[['tg','bg','pg','score','t_def','p_def']].mean().round(3))
print(f.score.describe().round(3))

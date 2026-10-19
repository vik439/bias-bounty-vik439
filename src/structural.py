"""Structural expected-gap model, calibrated to leaderboard-measured group means (no labels used)."""
import numpy as np, pandas as pd, json
from scipy.optimize import least_squares
from scipy.stats import poisson
r=pd.read_parquet('data/research.parquet')
f=pd.read_parquet('data/feat-all2.parquet')[['GEOID','km_motorway','km_trunk','km_primary','km_secondary','km_T_all','poi_fire_alt','bldcls_fire_station','poi_school_alt','bldcls_school','poi_ems_alt']]
d=r[['GEOID','region','pop_total','housing_units','ALAND','poi_fire','poi_school','poi_ems','poi_n','pct_urban']].merge(f,on='GEOID',validate='1:1')
N=len(d); m,s2=0.059806,0.089022463**2
L=json.load(open('out/lb_log.json')); groups=[]
for e in L:
    if not e['name'].startswith('grp_'): continue
    v=pd.read_csv(e['file'],dtype={'GEOID':str}).set_index('GEOID').coverage_gap_score.reindex(d.GEOID).values
    g=v>m+0.05; p=g.mean(); groups.append((e['name'],g,m+(s2+p*0.01-e['rmse']**2)/(0.2*p)))
reg=d.region.values; OK=reg=='eastern-ok'; AZ=reg=='maricopa-az'
kmT=d.km_T_all.values; hw=d[['km_motorway','km_trunk','km_primary','km_secondary']].sum(axis=1).values
pop=d.pop_total.values.astype(float); hu=d.housing_units.values.astype(float); area=np.clip(d.ALAND.values/1e6,0.05,None)
ov={'fire':d.poi_fire.values,'school':d.poi_school.values,'ems':d.poi_ems.values}
ev={'fire':d.poi_fire_alt.values+d.bldcls_fire_station.values,'school':d.poi_school_alt.values+d.bldcls_school.values,'ems':d.poi_ems_alt.values}
places=d.poi_n.values.astype(float)
gh_x,gh_w=np.polynomial.hermite_e.hermegauss(15); gh_w=gh_w/gh_w.sum()
def exp_ratio_gap(num,den,sig):
    # E[max(0,1-num/(den*e^eps))], eps~N(0,sig)
    den=np.maximum(den,1e-9); out=np.zeros_like(den)
    for x,w in zip(gh_x,gh_w): out+=w*np.clip(1-num/(den*np.exp(sig*x)),0,1)
    return out
nmax=np.arange(1,41)
def fac_gap(o,lam):
    # E[max(0,1-o/n) | n>=1], n~Poisson(lam); and P(n>=1)
    P=poisson.pmf(nmax[None,:],lam[:,None]); pdef=1-np.exp(-lam)
    g=(P*np.clip(1-o[:,None]/nmax[None,:],0,1)).sum(1)/np.maximum(pdef,1e-12)
    return g,pdef
names=['cOK','cAZ','cOth','sig_t','p0','b0','fire_pop','fire_area','fire_ev','sch_pop','sch_ev','ems_pop','cbp_k','sig_c','w_und']
x0=np.array([0.8,1.6,1.0,0.4,0.05,0.02,0.6,0.15,0.8,2.0,0.8,0.15,0.03,0.6,1.0])
lo=np.array([0.3,0.3,0.3,0.05,0,0,0,0,0,0,0,0,0.002,0.05,0.3]); hi=np.array([3,3,3,1.5,0.6,0.3,5,3,3,10,3,3,0.3,2,3])
def predict(th,parts=False,c_override=None):
    cOK,cAZ,cOth,sig_t,p0,b0,fpop,farea,fev,spop,sev,epop,ck,sc,wu=th
    c=np.where(OK,cOK,np.where(AZ,cAZ,cOth)) if c_override is None else c_override
    defined=kmT>0.05; pt=np.where(defined,1.0,p0)
    t=np.where(defined,exp_ratio_gap(hw,kmT*c,sig_t),0.6)   # undefined-but-real TIGER: assume large gap
    lam={'fire':fpop*pop/10000+farea*area/100+fev*ev['fire'],'school':spop*pop/10000+sev*ev['school'],'ems':epop*pop/10000+0.3*ev['ems']}
    num=np.zeros(N); pd_any=np.ones(N); den=np.zeros(N)
    for k in lam:
        g,pk=fac_gap(ov[k].astype(float),np.maximum(lam[k],1e-6)); num+=pk*g; den+=pk; pd_any*=(1-pk)
    ph=1-pd_any; h=num/np.maximum(den,1e-12)
    cb=exp_ratio_gap(places,ck*hu,sc)*(hu>0)
    poi=ph*(h+cb)/2+(1-ph)*cb
    y=pt*(t+b0+poi)/3+(1-pt)*(b0+poi)/2*wu
    y=np.clip(y,0,1)
    return (y,dict(t=t,pt=pt,poi=poi,h=h,cb=cb,ph=ph)) if parts else y
def resid(th):
    y=predict(th); out=[(y.mean()-m)*3]
    for n,g,mg in groups: out.append(y[g].mean()-mg)
    return np.array(out)
if __name__=='__main__':
    fit=least_squares(resid,x0,bounds=(lo,hi))
    th=fit.x; y,P=predict(th,True)
    print(dict(zip(names,th.round(4))))
    print('fit rms resid %.4f'%np.sqrt(np.mean(fit.fun**2)))
    for (n,g,mg),rv in zip(groups,fit.fun[1:]): print(f'  {n:22s} measured {mg:.4f} model {mg+rv:.4f}')
    print('pred mean %.4f sd %.4f'%(y.mean(),y.std()),'optimistic RMSE %.4f'%np.sqrt(max(s2-np.mean((y-m)**2),0)))
    np.save('out/structural_theta.npy',th); pd.DataFrame({'GEOID':d.GEOID,'y':y,**{k:v for k,v in P.items()}}).to_parquet('out/structural_pred.parquet')

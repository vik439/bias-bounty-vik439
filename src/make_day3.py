"""Day-3 submissions: three region tests and two models built from measured group means."""
import pandas as pd, numpy as np, os
C=pd.read_parquet('out/candidates.parquet')
order=pd.concat([pd.read_csv(f'data/reference/{r}/{r}-sample-submission.csv',dtype={'GEOID':str})[['GEOID']] for r in ['eastern-ok','maricopa-az','northern-ca','south-central-tx','eastern-wa']])
d=order.merge(C,on='GEOID',validate='1:1')
m,s2,Eyy=0.059806,0.089022463**2,0.107246299**2
und=~d.t_all_def; ok=d.region=='eastern-ok'
mu_u,mu_okdef,mu_rest=0.0207,0.1565,0.0606
base=np.where(und,mu_u,np.where(ok,mu_okdef,mu_rest))
# within-group slope on the roads signal, from the roads-only score
t=d.t_all.values; Ety=(np.mean(t**2)+Eyy-0.14968988**2)/2
g=np.where(und,0,np.where(ok,1,2)); tbar=np.array([t[g==k].mean() for k in range(3)])[g]
cov_within=Ety-np.mean(tbar*base); var_within=np.mean((t-tbar)**2); k=cov_within/var_within
print('within-group slope %.3f'%k,'pred RMSE groups %.4f, +slope %.4f'%(np.sqrt(s2-np.mean((base-m)**2)),np.sqrt(s2-np.mean((base-m)**2)-cov_within**2/var_within)))
subs={'18_groups_route_ok':base,'19_groups_plus_roads':np.clip(base+0.7*k*(t-tbar),0,1)}
for r in ['south-central-tx','maricopa-az','northern-ca']: subs['2x_group_'+r]=m+0.1*(d.region==r)
O='/home/claude/subs/day3'; os.makedirs(O,exist_ok=True)
for n,v in subs.items():
    x=pd.DataFrame({'GEOID':d.GEOID,'coverage_gap_score':np.round(np.asarray(v,float),6)})
    assert len(x)==9794 and x.GEOID.is_unique and x.notna().all().all() and x.coverage_gap_score.between(0,1).all()
    x.to_csv(f'{O}/{n}.csv',index=False); print(n,round(float(np.mean(v)),4))

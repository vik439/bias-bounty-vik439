import numpy as np, pandas as pd, os
from scipy.optimize import least_squares
import structural as S
O='/home/claude/subs/day6/'
th1=np.load('out/structural_theta.npy'); y1=S.predict(th1)
# variant 2: transport scale fixed to the organisers' stated regional ratios (c = 1 except Oklahoma 0.79), others refit
fix={0:0.79,1:1.0,2:1.0}
free=[i for i in range(len(S.x0)) if i not in fix]
def full(z):
    th=S.x0.copy(); th[free]=z
    for i,v in fix.items(): th[i]=v
    return th
fit=least_squares(lambda z:S.resid(full(z)),S.x0[free],bounds=(S.lo[free],S.hi[free]))
th2=full(fit.x); y2=S.predict(th2)
print('variant2',dict(zip(S.names,th2.round(4))),'rms %.4f'%np.sqrt(np.mean(fit.fun**2)))
m=S.m
for name,y in [('55_structural_A',y1),('56_structural_B_fixed_roads',y2)]:
    x=pd.DataFrame({'GEOID':S.d.GEOID,'coverage_gap_score':np.round(np.clip(y,0,1),6)})
    assert len(x)==9794 and x.GEOID.is_unique and x.notna().all().all(); x.to_csv(O+name+'.csv',index=False)
    print(name,'mean %.4f sd %.4f'%(y.mean(),y.std()))
v5=pd.read_csv('/home/claude/subs/day4/36_v5_tier_roads.csv',dtype={'GEOID':str}).set_index('GEOID').coverage_gap_score.reindex(S.d.GEOID).values
print('corr v5-A %.3f v5-B %.3f A-B %.3f'%(np.corrcoef(v5,y1)[0,1],np.corrcoef(v5,y2)[0,1],np.corrcoef(y1,y2)[0,1]))

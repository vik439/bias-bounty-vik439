import numpy as np, pandas as pd, sys
from scipy.optimize import least_squares
import structural as S
O='/home/claude/subs/day7/'
best=pd.read_csv('/home/claude/subs/day6/58_blend_v5_A_B.csv',dtype={'GEOID':str}).set_index('GEOID').coverage_gap_score.reindex(S.d.GEOID).values
B=pd.read_csv('/home/claude/subs/day6/56_structural_B_fixed_roads.csv',dtype={'GEOID':str}).coverage_gap_score.values
for tag,fix in [('63_structural_F_roads13',{0:1.0,1:1.3,2:1.3}),('64_structural_G_roads06',{0:0.6,1:0.6,2:0.6})]:
    free=[i for i in range(len(S.x0)) if i not in fix]
    def full(z):
        th=S.x0.copy(); th[free]=z
        for i,v in fix.items(): th[i]=v
        return th
    f=least_squares(lambda z:S.resid(full(z)),S.x0[free],bounds=(S.lo[free],S.hi[free]),max_nfev=40)
    y=S.predict(full(f.x))
    pd.DataFrame({'GEOID':S.d.GEOID,'coverage_gap_score':np.round(np.clip(y,0,1),6)}).to_csv(O+tag+'.csv',index=False)
    print(tag,'rms %.4f'%np.sqrt(np.mean(f.fun**2)),'corr best %.3f corr B %.3f'%(np.corrcoef(best,y)[0,1],np.corrcoef(B,y)[0,1]),flush=True)

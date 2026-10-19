"""Day-7 structural variants: C (per-region road scales), D (road part only), E (size-weighted fit)."""
import numpy as np, pandas as pd, os
from scipy.optimize import least_squares
import structural as S
O='/home/claude/subs/day7/'; os.makedirs(O,exist_ok=True)
reg=S.reg; R5=['eastern-ok','maricopa-az','northern-ca','south-central-tx','eastern-wa']
base=np.load('out/structural_theta.npy')
# C: replace the 3 road scales by 5 regional ones
def predC(z):
    th=base.copy(); th[3:]=z[5:]
    return S.predict(th,c_override=np.select([reg==r for r in R5],z[:5]))
def residC(z):
    y=predC(z); o=[(y.mean()-S.m)*3]+[y[g].mean()-mg for _,g,mg in S.groups]; return np.array(o)
z0=np.r_[base[0],base[1],base[2],base[2],base[2],base[3:]]
lo=np.r_[[0.3]*5,S.lo[3:]]; hi=np.r_[[3]*5,S.hi[3:]]
fC=least_squares(residC,z0,bounds=(lo,hi),max_nfev=40); yC=predC(fC.x); print('C rms %.4f'%np.sqrt(np.mean(fC.fun**2)),'scales',fC.x[:5].round(3))
# D: road part only, expected transport gap times definedness, rescaled to the target mean/spread
_,P=S.predict(base,True); yD=P['pt']*P['t']; yD=S.m+(yD-yD.mean())*0.03/yD.std()
# E: fit weighting each group by sqrt of its tract share (bigger groups count more)
w=np.array([3]+[np.sqrt(g.mean())*4 for _,g,_ in S.groups])
fE=least_squares(lambda th:S.resid(th)*w,base,bounds=(S.lo,S.hi),max_nfev=40); yE=S.predict(fE.x); print('E rms %.4f'%np.sqrt(np.mean(fE.fun**2/w**2)))
for n,y in [('60_structural_C_regional_roads',yC),('61_structural_D_roads_part',yD),('62_structural_E_weighted',yE)]:
    x=pd.DataFrame({'GEOID':S.d.GEOID,'coverage_gap_score':np.round(np.clip(y,0,1),6)})
    assert len(x)==9794 and x.GEOID.is_unique and x.notna().all().all(); x.to_csv(O+n+'.csv',index=False)
A=pd.read_csv('/home/claude/subs/day6/55_structural_A.csv',dtype={'GEOID':str}).coverage_gap_score.values
B=pd.read_csv('/home/claude/subs/day6/58_blend_v5_A_B.csv',dtype={'GEOID':str}).set_index('GEOID').coverage_gap_score.reindex(S.d.GEOID).values
for n,y in [('C',yC),('D',yD),('E',yE)]: print(n,'corr with A %.3f, with best blend %.3f'%(np.corrcoef(A,y)[0,1],np.corrcoef(B,y)[0,1]))

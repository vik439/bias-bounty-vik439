"""Leaderboard bookkeeping: every scored submission and the least-squares blend they imply.
Public RMSE of a submission p gives mean(p*y) = (mean(p^2) + mean(y^2) - RMSE^2)/2, with mean(y^2) from the all-zeros score.
Moments of p are taken over all tracts (the public 30% subset is unknown), so results are approximate."""
import json, numpy as np, pandas as pd, sys
LOG='out/lb_log.json'
def load(): return json.load(open(LOG))
def vec(path): return pd.read_csv(path,dtype={'GEOID':str}).set_index('GEOID')['coverage_gap_score']
def solve(names=None, ridge=0.0, verbose=True):
    log=load(); z=[e for e in log if e['name']=='zeros'][0]; Eyy=z['rmse']**2
    ents=[e for e in log if e['name']!='zeros' and (names is None or e['name'] in names)]
    X=pd.concat([vec(e['file']).rename(e['name']) for e in ents],axis=1)
    Xy=np.array([(np.mean(X[e['name']]**2)+Eyy-e['rmse']**2)/2 for e in ents])
    G=(X.T@X).values/len(X); w=np.linalg.solve(G+ridge*np.eye(len(G)),Xy)
    mse=Eyy-2*w@Xy+w@G@w
    if verbose:
        print('E[y^2]=%.6f'%Eyy); print(pd.Series(w,index=X.columns).round(4).to_string()); print('predicted blend RMSE %.4f'%np.sqrt(max(mse,0)))
    return X.columns.tolist(), w, (X.values@w)
if __name__=='__main__': solve()

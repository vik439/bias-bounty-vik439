"""After scoring, solve the exact public-set optimal blend of a small set of scored files (plus a constant)."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'src'); import lb
names=sys.argv[1].split(','); ridge=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
cols,w,p=lb.solve(['const010']+names,ridge=ridge)
X=pd.concat([lb.vec(e['file']).rename(e['name']) for e in lb.load() if e['name'] in ['const010']+names],axis=1)
out=pd.DataFrame({'GEOID':X.index,'coverage_gap_score':np.round(np.clip(p,0,1),6)})
assert len(out)==9794 and out.notna().all().all()
fn=sys.argv[3]; out.to_csv(fn,index=False); print('wrote',fn,'clipped share %.3f'%((p<0)|(p>1)).mean())

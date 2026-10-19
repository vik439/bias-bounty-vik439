import pandas as pd, glob
f=pd.concat([pd.read_parquet(p) for p in sorted(glob.glob('data/feat-*-*.parquet'))])
d=pd.read_parquet('data/routes-long.parquet')
f['km_T_all']=f.GEOID.map(d.groupby('GEOID')['km'].sum()).fillna(0)
f.to_parquet('data/feat-all.parquet'); print(f.groupby('region').size())

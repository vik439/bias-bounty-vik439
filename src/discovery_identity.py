"""Per-tract 'identified structure' measures from provided Overture layers (buildings: class/name; places: address)."""
import duckdb, pandas as pd, numpy as np, geopandas as gpd, pyarrow.parquet as pq
REG=['northern-ca','eastern-ok','maricopa-az','south-central-tx','eastern-wa']
out=[]
for r in REG:
    tr=gpd.read_parquet(f'data/strata/{r}/{r}-census-tracts.parquet')[['GEOID','geometry']].reset_index(drop=True)
    con=duckdb.connect(); con.sql("SET memory_limit='2GB'; SET threads=2")
    con.sql(f"""COPY (SELECT (bbox.xmin+bbox.xmax)/2 x,(bbox.ymin+bbox.ymax)/2 y, sources[1].dataset ds,
        (class IS NOT NULL)::INT has_class, (names.primary IS NOT NULL)::INT has_name
        FROM 'data/reference/{r}/{r}-overture-buildings.parquet') TO 'data/tmp-id.parquet'""")
    acc=[]
    for b in pq.ParquetFile('data/tmp-id.parquet').iter_batches(batch_size=500000):
        t=b.to_pandas(); g=gpd.GeoDataFrame(t,geometry=gpd.points_from_xy(t.x,t.y),crs=tr.crs)
        j=gpd.sjoin(g,tr,predicate='within',how='inner'); j=j[~j.index.duplicated()]
        j['ml']=j.ds.isin(['Microsoft ML Buildings','Google Open Buildings']).astype(int)
        acc.append(j.groupby('GEOID')[['has_class','has_name','ml']].sum().assign(n=j.groupby('GEOID').size()))
    a=pd.concat(acc).groupby(level=0).sum()
    p=con.sql(f"""SELECT (bbox.xmin+bbox.xmax)/2 x,(bbox.ymin+bbox.ymax)/2 y,
        (len(addresses)>0 AND addresses[1].freeform IS NOT NULL)::INT has_addr, categories.primary cat
        FROM 'data/reference/{r}/{r}-overture-pois.parquet'""").df()
    g=gpd.GeoDataFrame(p,geometry=gpd.points_from_xy(p.x,p.y),crs=tr.crs); j=gpd.sjoin(g,tr,predicate='within'); j=j[~j.index.duplicated()]
    a['poi']=j.groupby('GEOID').size(); a['poi_addr']=j.groupby('GEOID').has_addr.sum()
    out.append(a.fillna(0).assign(region=r)); print(r,len(a),flush=True)
pd.concat(out).reset_index().rename(columns={'index':'GEOID'}).to_parquet('out/identity.parquet')

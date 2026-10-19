"""Long table: km of route-tagged Overture road per tract x network-set x class."""
import numpy as np, pandas as pd, geopandas as gpd, pyarrow.parquet as pq, shapely
out=[]
for r in ['northern-ca','eastern-ok','maricopa-az','south-central-tx','eastern-wa']:
    tr=gpd.read_parquet(f'data/strata/{r}/{r}-census-tracts.parquet')[['GEOID','geometry']].to_crs(5070).reset_index(drop=True)
    for b in pq.ParquetFile(f'data/reference/{r}/{r}-overture-roads.parquet').iter_batches(batch_size=150000,columns=['class','routes','geometry']):
        df=b.to_pandas(); df=df[df['routes'].apply(lambda a: a is not None and len(a)>0)]
        if not len(df): continue
        df['net']=df['routes'].apply(lambda a: '|'.join(sorted({(q.get('network') or '?') for q in a})))
        g=gpd.GeoDataFrame(df[['class','net']],geometry=shapely.from_wkb(df['geometry'].values),crs='OGC:CRS84').to_crs(5070)
        j=gpd.sjoin(g,tr,predicate='intersects',how='inner')
        j['km']=shapely.length(shapely.intersection(j.geometry.values,tr.geometry.reindex(j['index_right']).values))/1000
        out.append(j.groupby(['GEOID','net','class'],as_index=False)['km'].sum().assign(region=r))
d=pd.concat(out).groupby(['region','GEOID','net','class'],as_index=False)['km'].sum()
d.to_parquet('data/routes-long.parquet'); print(d.shape)
for r,x in d.groupby('region'):
    print(r); print(x.groupby('net')['km'].sum().sort_values(ascending=False).head(25).round(0).to_string())

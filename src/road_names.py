"""Per-tract km of non-highway-class roads whose names look like numbered highways (possible TIGER S1100/S1200)."""
import numpy as np, pandas as pd, geopandas as gpd, pyarrow.parquet as pq, shapely, re
PAT=re.compile(r'\b(?:interstate|i-\s?\d|us\s?(?:highway|hwy|route|rte)?\s?\d|state\s?(?:highway|hwy|route|rte)|sh\s?\d|hwy|highway|fm\s?\d|farm(?: to|-to-)market|rm\s?\d|ranch(?: to|-to-)market|county\s?(?:road|rd|highway|hwy|route)|cr\s?\d|loop\s?\d|spur\s?\d|bia\s|route\s?\d|state route|sr\s?\d|ca-\d|az-\d|ok-\d|wa-\d|tx-\d)',re.I)
out=[]
for r in ['northern-ca','eastern-ok','maricopa-az','south-central-tx','eastern-wa']:
    tr=gpd.read_parquet(f'data/strata/{r}/{r}-census-tracts.parquet')[['GEOID','geometry']].to_crs(5070).reset_index(drop=True)
    for b in pq.ParquetFile(f'data/reference/{r}/{r}-overture-roads.parquet').iter_batches(batch_size=150000,columns=['class','names','geometry']):
        df=b.to_pandas(); nm=df['names'].apply(lambda x: (x or {}).get('primary') or '')
        df['hwname']=nm.str.contains(PAT); df=df[df.hwname | df['class'].isin(['tertiary'])]
        if not len(df): continue
        g=gpd.GeoDataFrame(df[['class','hwname']],geometry=shapely.from_wkb(df['geometry'].values),crs='OGC:CRS84').to_crs(5070)
        j=gpd.sjoin(g,tr,predicate='intersects',how='inner')
        j['km']=shapely.length(shapely.intersection(j.geometry.values,tr.geometry.reindex(j['index_right']).values))/1000
        j['hw']=j['class'].isin(['motorway','trunk','primary','secondary'])
        out.append(j.groupby(['GEOID','class','hwname'],as_index=False)['km'].sum())
    print(r,flush=True)
d=pd.concat(out).groupby(['GEOID','class','hwname'],as_index=False)['km'].sum(); d.to_parquet('data/road_names_long.parquet'); print(d.shape)

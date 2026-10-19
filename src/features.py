"""Per-tract features from the provided challenge layers only. Deterministic."""
import sys, numpy as np, pandas as pd, geopandas as gpd, pyarrow.parquet as pq, shapely
REGIONS=['northern-ca','eastern-ok','maricopa-az','south-central-tx']
HW=['motorway','trunk','primary','secondary']
SCHOOL=['elementary_school','middle_school','high_school','school','private_school','public_school']
def pts_to_tract(x,y,tracts):
    g=gpd.GeoDataFrame({'i':np.arange(len(x))},geometry=gpd.points_from_xy(x,y),crs=tracts.crs)
    j=gpd.sjoin(g,tracts[['GEOID','geometry']],predicate='within',how='inner')
    j=j[~j.index.duplicated()]
    out=pd.Series(index=np.arange(len(x)),dtype=object); out.loc[j['i'].values]=j['GEOID'].values
    return out.values
def batches(path,cols,n=150000):
    pf=pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=n,columns=cols):
        df=b.to_pandas(); df['geometry']=shapely.from_wkb(df['geometry'].values)
        yield gpd.GeoDataFrame(df,geometry='geometry',crs='OGC:CRS84')
def acc(f,name,ser):
    if name not in f: f[name]=0.0
    f[name]=f[name].add(ser,fill_value=0).reindex(f.index).fillna(0) if len(ser) else f[name]
def run(r):
    R=f'data/reference/{r}/{r}'; S=f'data/strata/{r}/{r}'
    sub=pd.read_csv(f'{R}-sample-submission.csv',dtype={'GEOID':str})
    tracts=gpd.read_parquet(f'{S}-census-tracts.parquet')[['GEOID','ALAND','geometry']]
    f=pd.DataFrame({'GEOID':sub.GEOID}).set_index('GEOID')
    tracts=tracts.reset_index(drop=True)
    # buildings
    import duckdb
    con=duckdb.connect(); con.sql("SET memory_limit='2GB'; SET threads=2; SET preserve_insertion_order=false")
    con.sql(f"COPY (SELECT (bbox.xmin+bbox.xmax)/2 x,(bbox.ymin+bbox.ymax)/2 y,sources[1].dataset ds,class FROM '{R}-overture-buildings.parquet') TO 'data/tmp-bld.parquet'"); con.close()
    for b in pq.ParquetFile('data/tmp-bld.parquet').iter_batches(batch_size=500000):
        t=b.to_pandas(); t['GEOID']=pts_to_tract(t.x.values,t.y.values,tracts)
        acc(f,'bld_n',t.groupby('GEOID').size())
        for k,v in {'bld_ms':'Microsoft ML Buildings','bld_osm':'OpenStreetMap','bld_esri':'Esri Community Maps'}.items():
            acc(f,k,t[t.ds==v].groupby('GEOID').size())
        for c in ['fire_station','school','hospital']:
            acc(f,'bldcls_'+c,t[t['class']==c].groupby('GEOID').size())
    # pois
    for p in batches(f'{R}-overture-pois.parquet',['categories','names','confidence','geometry']):
        p['cat']=p['categories'].str['primary']; p['alt']=p['categories'].str['alternate'].apply(lambda a: '|'.join(a) if a is not None else '')
        p['name']=p['names'].str['primary'].fillna('').str.lower()
        p['GEOID']=pts_to_tract(p.geometry.x.values,p.geometry.y.values,tracts)
        acc(f,'poi_n',p.groupby('GEOID').size())
        acc(f,'poi_n_c50',p[p.confidence>=0.5].groupby('GEOID').size())
        acc(f,'poi_fire',p[p.cat=='fire_department'].groupby('GEOID').size())
        acc(f,'poi_ems',p[p.cat=='ambulance_and_ems_services'].groupby('GEOID').size())
        acc(f,'poi_school',p[p.cat.isin(SCHOOL)].groupby('GEOID').size())
        acc(f,'poi_fire_alt',p[(p.cat!='fire_department')&(p.alt.str.contains('fire_department')|p.name.str.contains(r'fire (?:station|dep|dist|rescue|protection)|volunteer fire|\bvfd\b'))].groupby('GEOID').size())
        acc(f,'poi_school_alt',p[(~p.cat.isin(SCHOOL))&(p.alt.str.contains('school')|p.name.str.contains(r'\b(?:elementary|middle|high|junior high) school\b|\bisd\b'))].groupby('GEOID').size())
        acc(f,'poi_ems_alt',p[(p.cat!='ambulance_and_ems_services')&(p.alt.str.contains('ambulance')|p.name.str.contains(r'\bems\b|ambulance'))].groupby('GEOID').size())
    # infrastructure
    inf=gpd.read_parquet(f'{R}-overture-infrastructure.parquet',columns=['class','geometry'])
    c=inf.geometry.representative_point(); inf['GEOID']=pts_to_tract(c.x.values,c.y.values,tracts)
    f['infra_n']=inf.groupby('GEOID').size()
    # roads: exact length per tract in EPSG:5070
    tr=tracts[['GEOID','geometry']].to_crs(5070)
    for rd in batches(f'{R}-overture-roads.parquet',['class','routes','names','geometry']):
        rd['routed']=rd['routes'].apply(lambda a: a is not None and len(a)>0)
        rd['net']=rd['routes'].apply(lambda a: '|'.join(sorted({(q.get('network') or '') for q in a})) if a is not None and len(a) else '')
        rd['named']=rd['names'].str['primary'].notna()
        def lens(g,name):
            g=g.to_crs(5070)
            j=gpd.sjoin(g[['geometry']],tr,predicate='intersects',how='inner')
            tg=tr.geometry.reindex(j['index_right']).values
            L=shapely.length(shapely.intersection(j.geometry.values,tg))
            acc(f,name,pd.Series(L,index=j['GEOID'].values).groupby(level=0).sum()/1000)
        for cl in HW+['tertiary']: lens(rd[rd['class']==cl],'km_'+cl)
        hw=rd[rd['class'].isin(HW)]
        lens(hw[hw.named],'km_hw_named'); lens(hw[hw.routed],'km_hw_routed')
        lens(rd[rd.routed],'km_routed_all'); lens(rd[rd.routed&~rd['class'].isin(HW)],'km_routed_nonhw')
        for n in ['US:I','US:US']: lens(rd[rd.net.str.contains(n+r'(?:\||$)')],'km_net_'+n.replace(':','_'))
        lens(rd[rd.net.str.contains(r'US:[A-Z]{2}(?:\||$|:)')&~rd.net.str.contains(r'US:US')],'km_net_state')
        L=rd.to_crs(5070); c=L.geometry.representative_point().to_crs(tracts.crs)
        L['GEOID']=pts_to_tract(c.x.values,c.y.values,tracts); acc(f,'km_all',L.assign(l=L.length/1000).groupby('GEOID')['l'].sum())
    acs=pq.read_table(f'{R}-census-acs-housing.parquet',columns=['GEOID','housing_units']).to_pandas().set_index('GEOID')
    f['housing_units']=acs['housing_units']
    st=pd.read_parquet(f'{S}-strata-tract-table.parquet').set_index('GEOID')
    f=f.fillna(0).join(st.select_dtypes(include=[np.number,bool]),how='left')
    f['region']=r; f.reset_index().to_parquet(f'data/feat-{r}.parquet'); print(r,f.shape,flush=True)
for r in (sys.argv[1:] or REGIONS): run(r)

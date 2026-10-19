import pandas as pd, numpy as np, os
c=pd.read_parquet('out/components.parquet')
order=pd.concat([pd.read_csv(f'data/reference/{r}/{r}-sample-submission.csv',dtype={'GEOID':str})[['GEOID']] for r in ['eastern-ok','maricopa-az','northern-ca','south-central-tx','eastern-wa']])
c=order.merge(c,on='GEOID',how='left',validate='1:1')
O='/mnt/user-data/outputs/zindi-bias-bounty'; os.makedirs(O,exist_ok=True)
subs={'01_zeros':np.zeros(len(c)),'02_const_0.10':np.full(len(c),0.1),'03_buildings_only':c.bg.values,'04_roads_only':c.tg.values,'05_places_only':c.pg.values,'06_combined_v1':c.score.values}
for k,v in subs.items():
    d=pd.DataFrame({'GEOID':c.GEOID,'coverage_gap_score':np.round(v,6)})
    assert len(d)==9794 and d.GEOID.is_unique and d.notna().all().all() and d.coverage_gap_score.between(0,1).all() and d.GEOID.str.len().eq(11).all()
    d.to_csv(f'{O}/{k}.csv',index=False)
    print(k, round(d.coverage_gap_score.mean(),4), round(d.coverage_gap_score.std(),4), (d.coverage_gap_score>0).mean().round(3))

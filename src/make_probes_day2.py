"""Day-2 submissions (revised after reading the public discussion board)."""
import pandas as pd, numpy as np, os, shutil
C=pd.read_parquet('out/candidates.parquet'); f=pd.read_parquet('data/feat-all2.parquet')
order=pd.concat([pd.read_csv(f'data/reference/{r}/{r}-sample-submission.csv',dtype={'GEOID':str})[['GEOID']] for r in ['eastern-ok','maricopa-az','northern-ca','south-central-tx','eastern-wa']])
d=order.merge(C,on='GEOID',validate='1:1').merge(f[['GEOID','pct_urban','pop_total','poi_fire']],on='GEOID',validate='1:1')
rural=d.pct_urban<0.5; tier=rural&(d.pop_total>=1500)&(d.poi_fire==0)
print('tier n',tier.sum(),'rural n',rural.sum())
m=0.059806
subs={'12_tier3_public_levels':np.where(tier,0.131,np.where(rural,0.0967,0.0475)),
      '07_buildings_only':d.b_usfs.values,
      '13_group_no_route_road':m+0.1*(~d.t_all_def),
      '14_group_eastern_ok':m+0.1*(d.region=='eastern-ok'),
      '15_business_share':d.cbp_share2.values}
for O in ['/mnt/user-data/outputs/zindi-bias-bounty/day2','/home/claude/subs/day2']:
    shutil.rmtree(O,ignore_errors=True); os.makedirs(O)
    for k,v in subs.items():
        x=pd.DataFrame({'GEOID':d.GEOID,'coverage_gap_score':np.round(np.asarray(v,float),6)})
        assert len(x)==9794 and x.GEOID.is_unique and x.notna().all().all() and x.coverage_gap_score.between(0,1).all()
        x.to_csv(f'{O}/{k}.csv',index=False)
for k,v in subs.items(): print(k, round(float(np.mean(v)),4))

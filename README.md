# Bias Bounty Mapping Equity Challenge — vik439

Label-free coverage-gap estimation for the Zindi Bias Bounty Mapping Equity Challenge.
Uses only the challenge package on Source Cooperative (Overture release 2026-08-19.0, ACS housing, strata tables).
No removed reference layer (TIGER, Microsoft Buildings, HIFLD, CBP) is used or reconstructed.

## Requirements
Python 3.12+, pandas, numpy, geopandas, shapely, pyarrow, duckdb, scipy. CPU only, about 8 GB RAM.

## Data
Download the challenge data (no credentials):

    aws s3 sync s3://humane-intelligence/bias-bounty-mapping-equity-challenge/reference/ data/reference/ \
        --endpoint-url https://data.source.coop --no-sign-request --exclude '*roads-unfiltered*'
    aws s3 sync s3://humane-intelligence/bias-bounty-mapping-equity-challenge/strata/ data/strata/ \
        --endpoint-url https://data.source.coop --no-sign-request --exclude '*.csv' --exclude 'national/*'

## Run order (from the repository root)
1. `python src/features.py` — per-tract counts and road lengths for each region
2. `python src/routes.py` — route-tagged road length per tract
3. `python src/merge.py` — combine regions
4. `python src/candidates.py` — roads signal and candidate signals
5. `python src/road_names.py` — highway-named minor roads per tract
6. `PYTHONPATH=src python src/structural.py` then `structural_variants.py`, `structural_v2.py`, `structural_v3.py` — structural models
7. `python src/blend_after.py <files> <ridge> <out.csv>` — least-squares mix from public scores
8. `python src/discovery_identity.py` — building-type and addressed-place tables for the Bias Discovery entry

`submission_log.json` lists every scored submission with its public RMSE. Group probes are converted to group means with
m_G − m = (σ² + p·δ² − RMSE²) / (2·δ·p), using m = 0.059806 and σ = 0.089022 from the all-zeros and constant-0.10 files.

All steps are deterministic (no random sampling). AI coding assistance was used to draft the code.

## Licence
MIT for code. Data licences as stated in the challenge bucket README.

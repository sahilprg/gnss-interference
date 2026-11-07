# Second receiver network: OpenSky

The map is built from one community receiver network (adsb.lol). OpenSky Network has
its own receivers and its own decoding. Comparing the two on the same cells and windows
checks our input and data handling (feeders, reception, decoding, aggregation): a
jammed aircraft reports a low NIC to every receiver, so the networks should agree where
both see enough aircraft. It does not check spoofing. The check and its decision rule
were declared before any OpenSky data was queried (docs/evaluation_protocol.md,
"Second network check 1").

## Access

OpenSky's historical database (Trino) is free for non-commercial research after an
application; this project has access until 1 Dec 2026. Read OpenSky's guide first:
https://openskynetwork.github.io/opensky-api/trino.html. Every query here filters on the
`hour` partition, covers one hour, aggregates on the server and runs alone; a query over
5 minutes is reported, and `--by-region` splits each hour into one query per region.

```
pip install -e .[opensky]
gnssguard opensky-probe                     # one hour, Baltic: what NIC and HCR look like
gnssguard opensky-fetch --dates 2026-10-26,...,2026-11-04
gnssguard opensky-check --dates 2026-10-26,...,2026-11-04
```

The first query opens a browser to sign in to OpenSky (username in lowercase); no
password is stored in the project. Results go to `data/opensky/<day>/<hour>.parquet`
(not committed: OpenSky data is not redistributed) and the report to `reports/opensky/`;
the summary table is copied to `results/` by hand after the run.

## What is compared

Per 1-degree cell and 15-minute window, airborne at or above 3,000 ft: aircraft with a
NIC value, and those with NIC < 7 in at least half of their messages (as in our cells).
A cell-window counts when both networks have 3 or more such aircraft; OpenSky flags it
when at least 30% of its aircraft are degraded. Reported: coverage, the OpenSky flag rate
where our model says jamming and where it says normal (also for the rules), the rank
correlation of the degraded share, the calm control, and the largest disagreements.

Cite OpenSky as: M. Schäfer, M. Strohmeier, V. Lenders, I. Martinovic, M. Wilhelm,
"Bringing Up OpenSky: A Large-scale ADS-B Sensor Network for Research", IPSN 2014.

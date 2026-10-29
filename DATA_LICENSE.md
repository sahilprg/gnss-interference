# Data license

The code in this repository is MIT licensed (see `LICENSE`).

The input data comes from the [adsb.lol open data archive](https://www.adsb.lol/docs/open-data/historical/),
made available under the [Open Database License (ODbL) 1.0](http://opendatacommons.org/licenses/odbl/1.0/).
It includes data from adsb.lol feeders and partners, FlyItalyADSB and TheAirTraffic.com.

The map background (`src/gnssguard/viz/data/ne_50m.json.gz`) is Natural Earth 1:50m land
and country-border data, which is in the public domain
([naturalearthdata.com](https://www.naturalearthdata.com/about/terms-of-use/)), taken from
the world-atlas 2.0.2 package.

Any dataset derived from it and published by this project (processed Parquet files,
aggregated interference grids, labels) is a derivative database and is released under
ODbL 1.0 with this attribution. Published datasets drop or hash aircraft identifiers.

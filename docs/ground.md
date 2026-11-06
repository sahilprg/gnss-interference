# Ground check: EUREF reference stations

The map comes from aircraft. To check it with a sensor that shares nothing with this
pipeline, `gnssguard ground-check` compares it with permanent geodetic GNSS stations of
the EUREF network, which record every satellite's signal strength (C/N0) every 30 s and
publish daily files openly. The rule and what is reported were fixed before any station
file was downloaded (docs/evaluation_protocol.md, "Ground check 1").

## Run it (on a machine with normal internet access)

```bash
pip install "hatanaka>=2.8.1"        # reads the compressed station files
gnssguard ground-fetch --dates 2026-10-26,2026-10-27,2026-10-28,2026-10-29,2026-10-30,2026-10-31,2026-11-01,2026-11-02,2026-11-03,2026-11-04
gnssguard ground-check --dates 2026-10-26,2026-10-27,2026-10-28,2026-10-29,2026-10-30,2026-10-31,2026-11-01,2026-11-02,2026-11-03,2026-11-04
```

The download is small (a few MB per station and day, about 60 station-days), by FTP from
BKG first (its HTTPS downloads have been off since a June 2026 outage), and goes to
`data/ground/<day>/`. Stations are in `configs/ground_stations.csv`; a station a data
centre does not have is printed as missing. Results go to `reports/ground/`:
`ground_check.csv` (per station and pooled) and `ground_windows.csv` (every window).

## What it measures

* **Ground:** per station and 15-minute window, the median GPS L1 C/A C/N0 over all
  satellites and the number of satellites tracked, against the same window of the day
  on the other days. Ground-affected: at least 3 dB weaker or 30% fewer satellites.
* **Aircraft:** the share of judged cell-windows within 150 km labeled jamming.
* **Agreement:** how often the aircraft flag nearby when the ground is affected, against
  when it is not (ratio above 1: they agree in time), the same with the aircraft series
  shifted by whole days as a control, and the rank correlation of the C/N0 drop with
  the flagged share.

## How to read it

A ground antenna hears a jammer only within a few tens of kilometres and with a clear
line of sight; aircraft at altitude hear it from hundreds. A weak or absent match is
therefore expected and does not refute the map. A nine-month ground monitor in Gdynia
found no correlation with ADS-B events (Inside GNSS, 2025). A clear match at some
stations is independent physical support, not proof. Spoofing is not checked.

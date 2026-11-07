import pandas as pd

from gnssguard.network import opensky as osk

BOXES = {"baltic": (53.0, 60.5, 18.0, 28.5), "central_europe": (47.5, 51.5, 7.0, 13.0)}


def test_every_query_filters_on_one_hour():
    q = osk.hour_sql(1793016000, BOXES)
    assert "WHERE hour = 1793016000" in q and "position_data4" in q
    assert "nic < 7" in q and "alt >= 914" in q and "'central_europe'" in q
    p = osk.probe_sql(BOXES["baltic"], 1793016000)
    assert "WHERE hour = 1793016000" in p


class FakeClient:
    def __init__(self):
        self.sql = []

    def query(self, sql, cached=False):
        self.sql.append(sql)
        return pd.DataFrame({"region": ["baltic"], "cell_lat": [55], "cell_lon": [21],
                             "w": [1793016000], "os_ac": [5], "os_degraded": [4],
                             "os_nic0": [2], "os_msg": [300]})


def test_fetch_one_query_per_hour_and_resumes(tmp_path):
    c = FakeClient()
    written = osk.fetch(["2026-10-26"], BOXES, tmp_path, client=c, sleep_s=0)
    assert len(written) == 24 and len(c.sql) == 24
    assert all("WHERE hour = " in q for q in c.sql)
    assert osk.fetch(["2026-10-26"], BOXES, tmp_path, client=c, sleep_s=0) == []
    assert len(c.sql) == 24  # nothing queried again
    osk.fetch(["2026-10-27"], BOXES, tmp_path, by_region=True, client=c, sleep_s=0)
    assert len(c.sql) == 24 + 48
    df = osk.load(tmp_path, ["2026-10-26"])
    assert df["os_ac"].iloc[0] == 5 * 24  # same window from every fake hour, summed


def ours_and_os(n_jam, n_norm, os_jam_flag, os_norm_flag):
    w = pd.Timestamp("2026-10-26T12:00Z")
    ours, os_ = [], []
    for i in range(n_jam + n_norm):
        jam = i < n_jam
        lat, lon = 53 + i % 7, 18 + i // 7
        win = w + pd.Timedelta(minutes=15 * (i // 50))
        ours.append({"region": "baltic", "cell_lat": lat, "cell_lon": lon, "window": win,
                     "n_ac_usable": 6, "n_ac_degraded": 5 if jam else 0,
                     "model": "jamming" if jam else "normal",
                     "rules": "jamming" if jam else "normal"})
        flag = (i < os_jam_flag) if jam else (i - n_jam < os_norm_flag)
        os_.append({"region": "baltic", "cell_lat": lat, "cell_lon": lon, "window": win,
                    "os_ac": 6, "os_degraded": 4 if flag else 0, "os_nic0": 0, "os_msg": 100})
    ours.append({"region": "baltic", "cell_lat": 60, "cell_lon": 28, "window": w,
                 "n_ac_usable": 4, "n_ac_degraded": 0, "model": "normal", "rules": "normal"})
    return pd.DataFrame(ours), pd.DataFrame(os_)


def test_agreement_and_disagreement(tmp_path):
    ours, os_ = ours_and_os(n_jam=60, n_norm=60, os_jam_flag=58, os_norm_flag=1)
    s, v = osk.run(ours, os_, tmp_path)
    pooled = s[s["region"] == "hotspots pooled"].iloc[0]
    assert pooled["our_judged"] == 121 and pooled["both_judged"] == 120
    assert pooled["model_jam_os_flag"] == round(58 / 60, 3)
    assert v.startswith("agreement")
    assert (tmp_path / "opensky_disagreements.csv").exists()

    ours, os_ = ours_and_os(n_jam=60, n_norm=60, os_jam_flag=30, os_norm_flag=1)
    _, v = osk.run(ours, os_, tmp_path)
    assert v.startswith("no agreement")

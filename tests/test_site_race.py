"""The map shows what the reader chose last, even when answers arrive in reverse order
(review of 9 Nov 2026). Runs in a real browser; skipped where none is installed."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gnssguard.product.publish import update_index

ROOT = Path(__file__).resolve().parents[1]
A, B = "2026-11-03", "2026-11-04"


@pytest.fixture
def site(tmp_path):
    from gnssguard.product.site import build_site

    out = build_site(tmp_path, ROOT / "configs" / "regions.yaml", ["baltic", "punjab"])
    for day, note in ((A, "ONLY-ON-THE-3RD"), (B, "ONLY-ON-THE-4TH")):
        s = {"date": day, "published": True, "day_reasons": [], "needs_review": [],
             "regions": {r: {"status": "withheld", "reasons": [f"{note} {r}"]}
                         for r in ("baltic", "punjab")}}
        (Path(out) / "data" / day).mkdir(parents=True, exist_ok=True)
        (Path(out) / "data" / day / "summary.json").write_text(json.dumps(s), encoding="utf-8")
        update_index(out, s)
    return Path(out)


def _browser():
    sync = pytest.importorskip("playwright.sync_api")
    pw = sync.sync_playwright().start()
    try:
        return pw, pw.chromium.launch()
    except Exception as e:  # no browser installed (e.g. CI): skip, do not fail
        pw.stop()
        pytest.skip(f"no browser: {e}")


def test_slow_answer_for_an_earlier_choice_does_not_replace_the_newer_one(site):
    pw, browser = _browser()
    try:
        page = browser.new_page()
        leaflet = os.environ.get("GNSSGUARD_LEAFLET_DIR")  # offline copy, if given

        def serve(route):
            url = route.request.url
            if url.startswith("https://unpkg.com/leaflet@1.9.4/dist/") and leaflet:
                return route.fulfill(path=str(Path(leaflet) / url.split("/dist/")[1]))
            if not url.startswith("http://site.test/"):
                return route.continue_()
            path = site / url.split("http://site.test/")[1].split("#")[0]
            if not path.is_file():
                return route.fulfill(status=404, body="")
            if url.endswith(f"data/{A}/summary.json"):
                page.wait_for_timeout(1500)  # the answer for the earlier choice is slow
            route.fulfill(path=str(path))

        page.route("**/*", serve)
        page.goto(f"http://site.test/index.html#date={B}&region=baltic")
        if page.evaluate("typeof L === 'undefined'"):
            pytest.skip("the map library (unpkg) is not reachable; set GNSSGUARD_LEAFLET_DIR")
        page.wait_for_function("document.getElementById('status').textContent.length > 0")
        page.select_option("#date", A)  # choose the 3rd (slow) ...
        page.select_option("#date", B)  # ... then the 4th at once
        page.wait_for_timeout(2500)  # the 3rd's answer has arrived by now
        status = page.text_content("#status")
        assert "ONLY-ON-THE-4TH baltic" in status and "3RD" not in status
        page.click("button[data-region=punjab]")
        page.wait_for_function(
            "document.getElementById('status').textContent.includes('punjab')")
        assert "ONLY-ON-THE-4TH punjab" in page.text_content("#status")
    finally:
        browser.close()
        pw.stop()

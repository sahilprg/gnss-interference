"""Command line interface: `gnssguard <command>`.

The commands live in four modules, one per area:

* pipeline: ingest, coverage, cells, label, map, synth, train, evaluate, rolling
* labels: blind samples, the labeling tool and the pre-declared analyses
* checks: public events, incident pages, ground stations, OpenSky, benchmark
* product: publish, daily, health

Each module defines its `cmd_*` functions and a `register(sub)` that adds its parsers.

    gnssguard ingest --date 2026-10-26
    gnssguard ingest --date 2026-10-26 --max-traces 5000 --out data/sample   # quick sample
    gnssguard cells && gnssguard label
    gnssguard map --region baltic --date 2026-10-26  # HTML map in reports/maps/
"""

from __future__ import annotations

import argparse
import logging
import sys

from gnssguard.cli import checks, labels, pipeline, product

# help lists the commands in this order
MODULES = (pipeline, labels, checks, product)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gnssguard")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    for module in MODULES:
        module.register(sub)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

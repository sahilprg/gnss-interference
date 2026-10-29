"""Download-free ingestion of the adsb.lol daily archive."""

from gnssguard.ingest.pipeline import ingest_day, is_done

__all__ = ["ingest_day", "is_done"]

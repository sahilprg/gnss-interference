"""Read a split tar archive as one continuous stream, without saving it to disk.

adsb.lol publishes each day as a tar archive split into parts (.tar.aa, .tar.ab, ...).
`MultiPartStream` concatenates the parts on the fly so `tarfile` can read members
sequentially (mode "r|"). Interrupted HTTP downloads resume with a Range request.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Sequence

import requests

log = logging.getLogger(__name__)

USER_AGENT = "gnssguard/0.1 (research; github.com/sahilprg/gnss-interference)"


class MultiPartStream(io.RawIOBase):
    """Concatenate local files or HTTP(S) URLs into one read-only byte stream."""

    def __init__(
        self,
        sources: Sequence[str],
        *,
        retries: int = 6,
        timeout: float = 60.0,
        session: requests.Session | None = None,
    ):
        super().__init__()
        if not sources:
            raise ValueError("no sources")
        self.sources = list(sources)
        self.retries = retries
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.total_read = 0
        self._idx = -1
        self._fh = None  # local file handle or urllib3 raw response
        self._resp = None
        self._part_offset = 0
        self._part_size: int | None = None
        self._next_part()

    # -- io.RawIOBase interface -------------------------------------------------
    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while True:
            if self._fh is None:
                return 0
            n = self._read_current(b)
            if n:
                self._part_offset += n
                self.total_read += n
                return n
            self._check_part_complete()
            self._next_part()

    def close(self) -> None:
        self._close_current()
        super().close()

    # -- internals ----------------------------------------------------------------
    @staticmethod
    def _is_url(s: str) -> bool:
        return s.startswith(("http://", "https://"))

    def _close_current(self) -> None:
        if self._resp is not None:
            self._resp.close()
            self._resp = None
        elif self._fh is not None:
            self._fh.close()
        self._fh = None

    def _next_part(self) -> None:
        self._close_current()
        self._idx += 1
        self._part_offset = 0
        self._part_size = None
        if self._idx >= len(self.sources):
            return
        src = self.sources[self._idx]
        log.info("opening part %d/%d: %s", self._idx + 1, len(self.sources), src.rsplit("/", 1)[-1])
        if self._is_url(src):
            self._open_http(src, 0)
        else:
            self._fh = open(src, "rb")  # noqa: SIM115 - closed in _close_current

    def _open_http(self, url: str, offset: int) -> None:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        resp = self.session.get(url, stream=True, timeout=self.timeout, headers=headers)
        if offset and resp.status_code != 206:
            resp.close()
            raise OSError(f"server ignored Range request (HTTP {resp.status_code}) for {url}")
        resp.raise_for_status()
        length = resp.headers.get("Content-Length")
        if length is not None and self._part_size is None:
            self._part_size = int(length) + offset
        self._resp = resp
        self._fh = resp.raw

    def _read_current(self, b) -> int:
        src = self.sources[self._idx]
        if not self._is_url(src):
            return self._fh.readinto(b)
        for attempt in range(self.retries + 1):
            try:
                data = self._fh.read(len(b))
                b[: len(data)] = data
                if not data and self._part_size is not None and self._part_offset < self._part_size:
                    raise OSError("connection closed before end of part")
                return len(data)
            except (OSError, requests.RequestException) as e:
                if attempt == self.retries:
                    raise
                wait = min(60, 2**attempt)
                log.warning("read error at byte %d of %s (%s); retry in %ds",
                            self._part_offset, src.rsplit("/", 1)[-1], e, wait)
                time.sleep(wait)
                self._close_current()
                self._open_http(src, self._part_offset)
        return 0

    def _check_part_complete(self) -> None:
        if self._part_size is not None and self._part_offset != self._part_size:
            raise OSError(
                f"part {self.sources[self._idx]} ended at {self._part_offset} "
                f"of {self._part_size} bytes"
            )

"""The HTTP stream must survive dropped connections and resume with a Range request."""

import urllib3

from gnssguard.ingest.stream import MultiPartStream

DATA = bytes(range(256)) * 400  # 102,400 bytes


class FakeRaw:
    def __init__(self, data, fail_after=None, exc=None):
        self.data, self.pos, self.fail_after, self.exc = data, 0, fail_after, exc

    def read(self, n):
        if self.fail_after is not None and self.pos >= self.fail_after:
            raise self.exc
        chunk = self.data[self.pos:self.pos + n]
        self.pos += len(chunk)
        return chunk


class FakeResp:
    def __init__(self, data, offset, status, raw):
        self.status_code = status
        self.headers = {"Content-Length": str(len(data) - offset)}
        self.raw = raw

    def raise_for_status(self):
        if self.status_code >= 400:
            raise OSError(f"HTTP {self.status_code}")

    def close(self):
        pass


class FlakySession:
    """First connect fails; the first response drops mid-way with urllib3 ProtocolError."""

    def __init__(self):
        self.headers = {}
        self.calls = []

    def get(self, url, stream, timeout, headers):
        self.calls.append(headers.get("Range", "full"))
        if len(self.calls) == 1:
            raise urllib3.exceptions.NewConnectionError(None, "connect failed")
        offset = int(headers["Range"][6:-1]) if "Range" in headers else 0
        if len(self.calls) == 2:
            exc = urllib3.exceptions.ProtocolError("Connection broken: IncompleteRead")
            raw = FakeRaw(DATA[offset:], fail_after=40_000, exc=exc)
        else:
            raw = FakeRaw(DATA[offset:])
        return FakeResp(DATA, offset, 206 if offset else 200, raw)


def test_resumes_after_connect_failure_and_dropped_read():
    sess = FlakySession()
    s = MultiPartStream(["https://example.org/day.tar.aa"], session=sess, backoff=0.0)
    got = b""
    while chunk := s.read(8192):
        got += chunk
    assert got == DATA
    assert sess.calls[0] == "full" and sess.calls[1] == "full"
    assert sess.calls[2] == "bytes=40960-"  # resumed exactly where the drop happened

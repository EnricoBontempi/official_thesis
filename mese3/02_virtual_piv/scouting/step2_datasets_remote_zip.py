#!/usr/bin/env python3
"""Partial access to large remote ZIP archives through HTTP Range requests.

Purpose (step 2 scouting, mission 02): reproduce the partial accesses reported in
`step2_datasets.md` §4 without downloading whole archives. Used for:
  * Cardiff DynamicStudio archive (figshare file 49264399, 7 301 821 358 bytes);
  * Neal et al. PIV-UQ "Turbulent region.zip" (4TU file c06997bb-8935-4e32-bc03-dc85c629f7e3);
  * TUdatalib bitstreams (need --ua, the server rejects requests without a browser user agent).

Commands:
  list    URL                         list members (size, name) from the central directory
  extract URL REGEX OUTDIR [--max N]  extract members whose name matches REGEX (stored or deflated)
  nested  URL INNER OUTDIR --max N --mb M
          INNER is a member that is itself a DEFLATED zip: stream-decompress its first M MiB
          and extract up to N complete inner members found sequentially (works only when the
          inner local headers carry sizes, i.e. no data descriptors).

Options: --ua STRING  send this User-Agent (e.g. "Mozilla/5.0" for TUdatalib); default: none
         (figshare's S3 redirect refuses some user agents).

Dependencies: Python >= 3.9 (standard library) and the `curl` executable, which must honour
the HTTPS proxy settings of the environment. Run with `python3 -I`. Write OUTDIR outside the
repository: downloaded data are never committed.
"""
import os
import re
import struct
import subprocess
import sys
import zipfile
import zlib


class RemoteFile:
    """Minimal seekable read-only file over HTTP Range (curl, follows redirects)."""

    def __init__(self, url, ua=None, block=1 << 16):
        self.url, self.pos, self.block = url, 0, block
        self.ua = ["-A", ua] if ua else []
        self.nreq = self.nbytes = 0
        self._cs, self._cb = 0, b""
        h = subprocess.run(["curl", "-sL", *self.ua, "-m", "60", "-r", "0-0", "-D", "-", "-o", os.devnull, url],
                           capture_output=True, text=True).stdout
        sizes = re.findall(r"(?i)content-range:\s*bytes 0-0/(\d+)", h)
        if not sizes:
            raise OSError(f"server did not honour a range request:\n{h[:500]}")
        self.size = int(sizes[-1])

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        if not (self._cs <= self.pos and self.pos + n <= self._cs + len(self._cb)):
            end = min(self.pos + max(n, self.block), self.size) - 1
            self._cb = self.get_range(self.pos, end)
            self._cs = self.pos
        o = self.pos - self._cs
        d = self._cb[o:o + n]
        self.pos += len(d)
        return d

    def get_range(self, start, end):
        d = subprocess.run(["curl", "-sL", *self.ua, "-m", "900", "-r", f"{start}-{end}", self.url],
                           capture_output=True).stdout
        self.nreq += 1
        self.nbytes += len(d)
        return d

    def close(self):
        pass


def _safe(name):
    return re.sub(r"[^A-Za-z0-9#._-]+", "_", name)[-150:]


def cmd_list(f):
    z = zipfile.ZipFile(f)
    print("SIZE", f.size, "NMEMBERS", len(z.infolist()))
    for i in z.infolist():
        print(f"{i.file_size:>12} {i.compress_type} {i.filename}")


def cmd_extract(f, regex, outdir, nmax):
    z = zipfile.ZipFile(f)
    os.makedirs(outdir, exist_ok=True)
    k = 0
    for i in z.infolist():
        if i.filename.endswith("/") or not re.search(regex, i.filename):
            continue
        p = os.path.join(outdir, _safe(i.filename))
        with open(p, "wb") as fh:
            fh.write(z.read(i))
        print("EXTRACTED", i.filename, "->", p, i.file_size)
        k += 1
        if nmax and k >= nmax:
            break


def cmd_nested(f, inner, outdir, nmax, mb):
    z = zipfile.ZipFile(f)
    zi = z.getinfo(inner)
    if zi.compress_type != zipfile.ZIP_DEFLATED:
        raise SystemExit("inner member is not deflated; use `extract` and open it locally")
    f.seek(zi.header_offset)
    h = f.read(30)
    n, m = struct.unpack("<HH", h[26:30])
    start = zi.header_offset + 30 + n + m
    raw = f.get_range(start, start + min(mb << 20, zi.compress_size) - 1)
    d = zlib.decompressobj(-15).decompress(raw)
    os.makedirs(outdir, exist_ok=True)
    p = k = 0
    while p + 30 <= len(d) and d[p:p + 4] == b"PK\x03\x04" and k < nmax:
        flags, method = struct.unpack("<HH", d[p + 6:p + 10])
        csize, usize, nl, xl = struct.unpack("<IIHH", d[p + 18:p + 30])
        name = d[p + 30:p + 30 + nl].decode("latin1")
        a = p + 30 + nl + xl
        if flags & 8 or a + csize > len(d):
            print("STOP at", name, "(data descriptor or incomplete stream)")
            break
        data = d[a:a + csize]
        data = zlib.decompress(data, -15) if method == 8 else data
        out = os.path.join(outdir, _safe(os.path.basename(inner) + "__" + name))
        with open(out, "wb") as fh:
            fh.write(data)
        print("EXTRACTED", inner, "::", name, "->", out, len(data))
        k += 1
        p = a + csize


def main():
    args = sys.argv[1:]
    ua = None
    if "--ua" in args:
        i = args.index("--ua"); ua = args[i + 1]; del args[i:i + 2]
    nmax = mb = 0
    if "--max" in args:
        i = args.index("--max"); nmax = int(args[i + 1]); del args[i:i + 2]
    if "--mb" in args:
        i = args.index("--mb"); mb = int(args[i + 1]); del args[i:i + 2]
    if not args or args[0] not in ("list", "extract", "nested"):
        print(__doc__); sys.exit(2)
    f = RemoteFile(args[1], ua=ua)
    if args[0] == "list":
        cmd_list(f)
    elif args[0] == "extract":
        cmd_extract(f, args[2], args[3], nmax)
    else:
        cmd_nested(f, args[2], args[3], nmax or 1, mb or 2)
    print("HTTP_RANGE_REQUESTS", f.nreq, "BYTES", f.nbytes, file=sys.stderr)


if __name__ == "__main__":
    main()

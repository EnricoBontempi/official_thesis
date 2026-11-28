#!/usr/bin/env python3
"""F2.5 - fetch the Cardiff members authorised by roadmap sec. 10.2 with ONE HTTP Range request per member.

The central directory is read through the scout's RemoteFile (tool copied from piv/step2-dataset-scouting @ ee9948e).
Each member is then fetched with a single range request (local header + compressed data), inflated, and verified against the
central-directory CRC-32 and size. Output goes OUTSIDE the repository. A provenance manifest (TSV) is written next to the data.
Run:  cd /tmp && python3 -I /path/to/f25_fetch.py
"""
import concurrent.futures as cf, datetime, hashlib, importlib.util, os, re, struct, sys, time, zipfile, zlib

URL = "https://ndownloader.figshare.com/files/49264399"
TOOLS = os.path.expanduser("~/piv_qualification_data/tools")
DEST = os.path.expanduser("~/piv_qualification_data/cardiff_d2017/data")
TOOLS_COMMIT = "ee9948e227f0d9272ae064cc8601c5e5399c28b4"
REGEX = (r"(I\.4xd8gr7l/0h6c/(data/AdaptivePIV#[0-9]+\.vectors|bled/data/ValRange#[0-9]+\.vectors|bled/h4h3/data/ValMovAvg#[0-9]+\.vectors|"
         r"bled/h4h3/99f/data/StatisticsVec#0\.vectors)|I\.4xd8gr7l/(.*/)?AnalysisEnsemble\.xml|D\.4xd8dily/(AnalysisEnsemble\.xml|data/MaskDefine#0\.image)|"
         r"20170303_143734/Run\.xml|1\.4wewffbs/AcquiredImageEnsemble\.xml|d\.2017\.0011500240/README\.txt)$")
spec = importlib.util.spec_from_file_location("rz", os.path.join(TOOLS, "step2_datasets_remote_zip.py"))
rz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rz)


def _range(start, end, tries=6):
    """One Range request via curl; backoff on HTTP 429/5xx. Returns bytes or None."""
    import subprocess
    for t in range(tries):
        p = subprocess.run(["curl", "-sL", "-m", "300", "-r", f"{start}-{end}", "-w", "\n%{http_code}", URL], capture_output=True)
        body, _, code = p.stdout.rpartition(b"\n")
        if code.strip() in (b"206", b"200"):
            return body
        time.sleep(min(120, 8 * 2 ** t))   # polite exponential backoff (rate limiting, not a data discrepancy)
    return None


def fetch(info):
    out_path = os.path.join(DEST, *info.filename.split("/"))
    if os.path.exists(out_path) and os.path.getsize(out_path) == info.file_size:
        data = open(out_path, "rb").read()
        if (zlib.crc32(data) & 0xFFFFFFFF) == info.CRC:
            return dict(path=info.filename, bytes=len(data), crc32=f"{info.CRC:08x}", crc_ok=True, sha256=hashlib.sha256(data).hexdigest(), attempts=0,
                        utc=datetime.datetime.fromtimestamp(os.path.getmtime(out_path), datetime.timezone.utc).isoformat(timespec="seconds"), note="reused (verified in an earlier run)")
    raw = _range(info.header_offset, info.header_offset + 30 + len(info.filename.encode()) + 400 + info.compress_size - 1)
    if raw is None or raw[:4] != b"PK\x03\x04":
        return dict(path=info.filename, bytes=info.file_size, crc32=f"{info.CRC:08x}", crc_ok=False, sha256="", attempts=6, utc="", note="request failed (rate limit?)")
    nl, xl = struct.unpack("<HH", raw[26:30])
    start = 30 + nl + xl
    comp = raw[start:start + info.compress_size]
    data = zlib.decompress(comp, -15) if info.compress_type == 8 else comp
    if (zlib.crc32(data) & 0xFFFFFFFF) == info.CRC and len(data) == info.file_size:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as fh:
            fh.write(data)
        return dict(path=info.filename, bytes=len(data), crc32=f"{info.CRC:08x}", crc_ok=True, sha256=hashlib.sha256(data).hexdigest(), attempts=1,
                    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), note="")
    return dict(path=info.filename, bytes=info.file_size, crc32=f"{info.CRC:08x}", crc_ok=False, sha256="", attempts=1, utc="", note="CRC/size MISMATCH")


if __name__ == "__main__":
    t0 = time.time()
    f = rz.RemoteFile(URL)
    z = zipfile.ZipFile(f)
    infos = [i for i in z.infolist() if not i.filename.endswith("/") and re.search(REGEX, i.filename)]
    print("archive bytes", f.size, "members", len(z.infolist()), "selected", len(infos), "selected bytes", sum(i.file_size for i in infos), flush=True)
    start = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rows = []
    with cf.ThreadPoolExecutor(2) as ex:
        for k, r in enumerate(ex.map(fetch, infos)):
            rows.append(r)
            if k % 100 == 0:
                print(k, f"{time.time() - t0:.0f}s", flush=True)
    end = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with open(os.path.join(os.path.dirname(DEST), "download_manifest_cardiff.tsv"), "w") as fh:
        fh.write("dataset\tdoi_version\turl\tdownload_utc\tmethod\tpath_in_archive\tbytes\tmd5_declared_vs_computed\tcrc32\tcrc_ok\tsha256\ttools_commit\tnote\n")
        for r in rows:
            fh.write(f"Cardiff d.2017.0011500240\t10.17035/d.2017.0011500240 (figshare 27051964 v1)\t{URL}\t{r['utc']}\tHTTP Range (1 request per member)\t"
                     f"{r['path']}\t{r['bytes']}\tn/a (archive not downloaded; declared a0dc9d9574a1ffdf14cc9634b97ddd52 refers to the whole zip)\t{r['crc32']}\t{r['crc_ok']}\t{r['sha256']}\t{TOOLS_COMMIT}\t{r['note']}\n")
    bad = [r for r in rows if not r["crc_ok"]]
    print("done", f"{time.time() - t0:.0f}s", "files", len(rows), "crc failures", len(bad), flush=True)
    if bad:
        print("FAILED:", [b["path"] for b in bad][:10])

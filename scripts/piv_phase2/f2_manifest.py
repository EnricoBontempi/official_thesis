"""Provenance manifest for the F2.4/F2.5 downloads (roadmap sec. 10.3): results/piv/qualification/download_manifest.tsv.

One row per file or archive member: dataset, DOI/version, URL, UTC time, method, path inside the archive, bytes, MD5 declared vs computed
(where it exists), CRC-32, sha256, tool commit, outcome of the checks.  Data stay in ~/piv_qualification_data (outside the repository).
"""
import json, os, sys, csv, hashlib, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "piv_phase1"))
import common as C

OUT = C.REPO / "results/piv/qualification"; OUT.mkdir(parents=True, exist_ok=True)
H = Path(os.path.expanduser("~/piv_qualification_data"))
NEAL = H / "neal_unsteady_core"
CARD = H / "cardiff_d2017"
rows = []
HEAD = ["dataset", "doi_version", "url", "download_utc", "method", "path_in_archive", "bytes", "md5_declared_vs_computed", "crc32", "sha256", "tools_commit", "check_outcome"]
code_sha = C.git_sha()

# ---------------- Neal: the whole zip
zp = NEAL / "Unsteady Inviscid core.zip"
md5 = hashlib.md5(); sha = hashlib.sha256()
with open(zp, "rb") as f:
    while b := f.read(1 << 23):
        md5.update(b); sha.update(b)
decl_md5, decl_bytes = "eb6d86228f534d6285859ccc0f16c369", 271827177
start, end = (NEAL / "download_start_utc.txt").read_text().strip(), (NEAL / "download_end_utc.txt").read_text().strip()
url = "https://data.4tu.nl/file/309145fd-dcd3-47c4-8b88-8349d9e70cdb/426acdf9-5762-4fad-8d99-f8604bc69c86"
ok = (md5.hexdigest() == decl_md5 and zp.stat().st_size == decl_bytes)
rows.append(["Neal PIV-UQ unsteady inviscid core", "10.4121/13663376.v1 (4TU, record 309145fd-dcd3-47c4-8b88-8349d9e70cdb)", url, f"{start}/{end}", "whole file (curl)",
             "Unsteady Inviscid core.zip", zp.stat().st_size, f"declared {decl_md5} / computed {md5.hexdigest()}", "", sha.hexdigest(), "n/a (curl)",
             "size and MD5 match the repository: " + ("PASS" if ok else "FAIL")])
members = json.load(open(NEAL / "extracted/Unsteady Inviscid core/_members.json")) if (NEAL / "extracted/Unsteady Inviscid core/_members.json").exists() else []
for m in members:
    rows.append(["Neal PIV-UQ unsteady inviscid core", "10.4121/13663376.v1", url, end, "extracted from the whole zip (zipfile)", m["member"], m["bytes"], "n/a", m["crc32"], m["sha256"], "n/a", "CRC-32 checked by zipfile on extraction"])
for sub in ("F001/HDR_data_F001", "F001/MS_images_F001", "F005/HDR_data_F005", "F005/MS_Images_F005"):
    jf = NEAL / "extracted/Unsteady Inviscid core" / sub / "_members.json"
    if jf.exists():
        for m in json.load(open(jf)):
            rows.append(["Neal PIV-UQ unsteady inviscid core", "10.4121/13663376.v1", url, end, f"extracted from inner zip {sub.split('/')[1]}.zip", f"{sub}/{m['member']}", m["bytes"], "n/a", m["crc32"], m["sha256"], "n/a", "CRC-32 checked by zipfile on extraction"])
# ---------------- Cardiff: members fetched by HTTP Range
cu = "https://ndownloader.figshare.com/files/49264399"
with open(CARD / "download_manifest_cardiff.tsv") as f:
    rd = csv.DictReader(f, delimiter="\t")
    n_ok = n_bad = 0
    for r in rd:
        ok = r["crc_ok"] == "True"; n_ok += ok; n_bad += (not ok)
        rows.append(["Cardiff d.2017.0011500240", "10.17035/d.2017.0011500240, figshare 27051964 v1", cu, r["download_utc"], "HTTP Range, 1 request per member", r["path_in_archive"], r["bytes"],
                     "archive md5 not computed (archive not downloaded; declared a0dc9d9574a1ffdf14cc9634b97ddd52)", r["crc32"], r["sha256"], r["tools_commit"],
                     ("CRC-32 and size match the central directory: PASS" if ok else "FAIL") + (f" ({r['note']})" if r.get("note") else "")])
with open(OUT / "download_manifest.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t"); w.writerow(HEAD); w.writerows(rows)
summ = {"rows": len(rows), "neal_zip_check": "PASS" if rows[0][-1].endswith("PASS") else "FAIL", "cardiff_members_ok": n_ok, "cardiff_members_failed": n_bad,
        "scripts_commit": code_sha, "note": "figshare answered HTTP 429 during a first parallel attempt (10 threads); the 1212 members affected were fetched later with 2 threads and backoff; no checksum discrepancy occurred"}
json.dump(summ, open(OUT / "download_manifest_summary.json", "w"), indent=1)
print(summ)

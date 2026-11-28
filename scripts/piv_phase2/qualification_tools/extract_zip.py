import sys, zipfile, os, hashlib, json
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
z = zipfile.ZipFile(src)
rec = []
for i in z.infolist():
    z.extract(i, dst)   # zipfile sanitises absolute paths and '..'
    if not i.is_dir():
        p = os.path.join(dst, i.filename)
        rec.append({"member": i.filename, "bytes": i.file_size, "crc32": f"{i.CRC:08x}", "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()})
json.dump(rec, open(os.path.join(dst, "_members.json"), "w"), indent=1)
print(len(rec), "files extracted")

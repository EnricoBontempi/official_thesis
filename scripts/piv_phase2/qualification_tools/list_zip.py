import sys, zipfile, collections
z = zipfile.ZipFile(sys.argv[1])
names = z.infolist()
print(len(names), "members; bad CRC member:", z.testzip())
dirs = collections.Counter("/".join(i.filename.split("/")[:-1]) for i in names)
for d, n in sorted(dirs.items())[:40]: print(n, d)
for i in names[:12]: print(i.file_size, i.filename)
print("ext:", collections.Counter(i.filename.rsplit(".", 1)[-1].lower() for i in names if "." in i.filename))

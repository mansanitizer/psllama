#!/usr/bin/env python3
"""Upload single files to the PS5 app directory and verify by SHA-256 readback.
usage: stage_file.py LOCAL_PATH..."""
import ftplib, hashlib, os, sys, time
for path in sys.argv[1:]:
    name = os.path.basename(path)
    dst = "/data/homebrew/PPSA99005/" + name
    f = ftplib.FTP(); f.connect("ps5.tailnet.example", 2121, timeout=120); f.login()
    local = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""): local.update(chunk)
    t = time.time()
    with open(path, "rb") as fh:
        f.storbinary("STOR " + dst, fh, blocksize=1 << 20)
    print(name, "uploaded in %.0fs" % (time.time() - t), flush=True)
    remote = hashlib.sha256(); f.retrbinary("RETR " + dst, remote.update, blocksize=1 << 20)
    print(name, "OK" if remote.hexdigest() == local.hexdigest() else "MISMATCH", flush=True)
    f.quit()
print("STAGED")

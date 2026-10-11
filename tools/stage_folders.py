#!/usr/bin/env python3
"""Upload model folders from ~/models/sd/<folder> to the PS5 app directory and
verify each file by reading it back (SHA-256). usage: stage_folders.py FOLDER..."""
import ftplib, hashlib, os, sys
for folder in sys.argv[1:]:
    src = os.path.expanduser(f"~/models/sd/{folder}")
    dst = f"/data/homebrew/PPSA99005/{folder}"
    f = ftplib.FTP(); f.connect("ps5.tailnet.example", 2121, timeout=120); f.login()
    try: f.mkd(dst)
    except ftplib.error_perm: pass
    for name in sorted(os.listdir(src)):
        if name.endswith(".aria2"): continue
        local = hashlib.sha256()
        with open(os.path.join(src, name), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""): local.update(chunk)
        with open(os.path.join(src, name), "rb") as fh:
            f.storbinary(f"STOR {dst}/{name}", fh, blocksize=1 << 20)
        remote = hashlib.sha256(); f.retrbinary(f"RETR {dst}/{name}", remote.update, blocksize=1 << 20)
        print(f"{folder}/{name}", "OK" if remote.hexdigest() == local.hexdigest() else "MISMATCH", flush=True)
    f.quit()
print("STAGED")

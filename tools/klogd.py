#!/usr/bin/env python3
"""The only reader of the PS5's klogsrv (it splits its stream between
concurrent readers). Appends every line, prefixed with the local time, to
~/ps5lab/klog/YYYY-MM-DD.log; everything else reads those files."""
import os, socket, sys, time

HOST = os.environ.get("PS5_HOST", "ps5.tailnet.example")
PORT = 3232
LOG_DIR = os.path.expanduser("~/ps5lab/klog")


def stamp():
    t = time.time()
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t % 1 * 1000):03d} "


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    day, out = None, None
    while True:
        try:
            s = socket.create_connection((HOST, PORT), timeout=10)
            s.settimeout(30)
            pending = b""
            idle = 0
            while True:
                try:
                    data = s.recv(65536)
                    idle = 0
                except socket.timeout:
                    # klogsrv is never silent for long; a connection quiet for
                    # 2 minutes is dead (e.g. the console went offline).
                    idle += 1
                    if idle >= 4:
                        break
                    continue
                if not data:
                    break
                pending += data
                *lines, pending = pending.split(b"\n")
                today = time.strftime("%Y-%m-%d")
                if today != day:
                    if out:
                        out.close()
                    day, out = today, open(os.path.join(LOG_DIR, today + ".log"), "a", buffering=1)
                for line in lines:
                    out.write(stamp() + line.decode("utf-8", "replace").rstrip("\r") + "\n")
        except OSError as error:
            sys.stderr.write(f"{stamp()}klogd: {error}; reconnecting\n")
        time.sleep(2)


if __name__ == "__main__":
    main()

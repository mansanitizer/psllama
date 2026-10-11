#!/usr/bin/env python3
"""After a PS5 reboot and re-jailbreak: start Tailscale and the kernel-log
service through the console's ELF loader on the direct cable, then restart the
log recorder. Safe to run when they are already up."""
import ftplib, io, os, socket, subprocess, time
LAN = os.environ.get("PS5_LAN_HOST", "10.42.0.211")          # the console on the direct cable
TAILNET = os.environ.get("PS5_HOST", "ps5.tailnet.example")   # its tailnet address
def port_open(host, port):
    try: socket.create_connection((host, port), timeout=3).close(); return True
    except OSError: return False
def load(data):
    s = socket.create_connection((LAN, 9021), timeout=15); s.sendall(data); s.shutdown(socket.SHUT_WR); s.close()
if port_open(TAILNET, 2121):
    print("Tailscale already up")
else:
    load(open(os.path.expanduser("~/ps5lm-inference/tailscale/tailscale-0.7.1.elf"), "rb").read())
    for _ in range(20):
        time.sleep(3)
        if port_open(TAILNET, 2121): break
    print("Tailscale", "up" if port_open(TAILNET, 2121) else "NOT up")
if port_open(TAILNET, 3232):
    print("klogsrv already up")
else:
    f = ftplib.FTP(); f.connect(LAN, 2121, timeout=20); f.login()
    b = io.BytesIO(); f.retrbinary("RETR /data/pldmgr/payloads/klogsrv/klogsrv_v0.9.elf", b.write); f.quit()
    load(b.getvalue()); time.sleep(4)
    print("klogsrv", "up" if port_open(TAILNET, 3232) else "NOT up")
subprocess.run(["systemctl", "--user", "restart", "ps5-klogd.service"])
time.sleep(8)
print(subprocess.run(["python3", os.path.expanduser("~/ps5lab/ps5ctl.py"), "status"], capture_output=True, text=True).stdout[:700])

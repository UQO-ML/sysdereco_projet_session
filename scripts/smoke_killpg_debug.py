"""Debug : le residuel pid est-il un zombie ou un process vivant ?"""
import os
import signal
import subprocess
import sys
import time

script = (
    "import subprocess, time\n"
    "child = subprocess.Popen(['python', '-c', "
    "'import time\\nwhile True: time.sleep(0.5)'])\n"
    "while True: time.sleep(0.5)\n"
)
proc = subprocess.Popen(
    ["python", "-c", script],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    start_new_session=True,
)
time.sleep(1.5)
pgid = os.getpgid(proc.pid)
print(f"parent pid={proc.pid}, pgid={pgid}")

# Dump avant kill
r = subprocess.run(["ps", "-e", "-o", "pid,ppid,pgid,stat,comm"], capture_output=True, text=True)
print("=== avant ===")
for line in r.stdout.splitlines()[0:1] + [l for l in r.stdout.splitlines() if f" {pgid} " in l]:
    print("  ", line)

# SIGKILL direct au groupe
os.killpg(pgid, signal.SIGKILL)
time.sleep(1.5)

r = subprocess.run(["ps", "-e", "-o", "pid,ppid,pgid,stat,comm"], capture_output=True, text=True)
print("=== apres SIGKILL immediat ===")
hits = [l for l in r.stdout.splitlines() if f" {pgid} " in l]
for line in r.stdout.splitlines()[0:1] + hits:
    print("  ", line)

# Interprete le STAT (3e lettre) :
# Z = zombie (defunct, en attente de reap par parent=init)
# R = running, S = sleep, T = stopped
for line in hits:
    parts = line.split(None, 4)
    if len(parts) >= 4:
        stat = parts[3]
        if stat.startswith("Z"):
            print(f"  -> pid={parts[0]} est un ZOMBIE (attend reap par init), pas un leak.")
        else:
            print(f"  -> pid={parts[0]} STAT={stat} <= process VIVANT, leak reel !")

# Reap manuel si necessaire (proc.wait ne voit que proc.pid=parent)
try:
    proc.wait(timeout=2)
except Exception:
    pass
print("proc.poll()=", proc.poll())

"""Smoke test du kill_vllm_proc_tree : simule vLLM avec un arbre de 3 process.

Verifie que `kill_vllm_proc_tree` tue bien tous les descendants (y compris les
petits-enfants) via `killpg` au lieu d'un simple `proc.terminate()` qui laisse
les orphelins derriere.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, "/workspace")
from src.ecr.scorer import kill_vllm_proc_tree

# Mock hierarchie : python level0 -> python level1 (enfant) -> python level2 (petit-enfant)
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
time.sleep(1.5)  # laisse enfants demarrer

pgid = os.getpgid(proc.pid)
print(f"parent pid={proc.pid}, pgid={pgid}")

r = subprocess.run(["ps", "-e", "-o", "pid,pgid,comm"], capture_output=True, text=True)
before = [l for l in r.stdout.splitlines() if f" {pgid} " in l]
print(f"=== avant kill : {len(before)} processus dans pgid={pgid} ===")
for line in before:
    print("  ", line)

kill_vllm_proc_tree(proc, timeout=5)
time.sleep(1.5)

r = subprocess.run(["ps", "-e", "-o", "pid,pgid,stat,comm"], capture_output=True, text=True)
group_lines = [l for l in r.stdout.splitlines() if f" {pgid} " in l]
# Separer zombies (morts, juste pas reap'es) des vivants.
alive = []
zombies = []
for line in group_lines:
    parts = line.split()
    # pid, pgid, stat, comm (stat est le 3e champ apres avoir split)
    stat = parts[2] if len(parts) >= 3 else ""
    if stat.startswith("Z"):
        zombies.append(line)
    else:
        alive.append(line)
print(f"=== apres kill : {len(alive)} vivants, {len(zombies)} zombies dans pgid={pgid} ===")
for line in group_lines:
    print("  ", line)

if alive:
    print("FAIL: des descendants VIVANTS ont survecu (leak reel)")
    sys.exit(1)
print("OK: tous les descendants vivants ont ete tues (zombies OK, init les reapera).")

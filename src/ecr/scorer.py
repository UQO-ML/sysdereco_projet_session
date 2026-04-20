"""Scorer subjectif local via vLLM + Qwen2.5-32B-Instruct-AWQ.

Remplace le scorer GPT-4-turbo de l'article (Section 5.6 + Appendix E).
Prompt repris du template Appendix E :

    You are evaluating the quality of a recommender system reply. Given the
    dialogue history and the reply, rate the reply on a 0-9 scale across 5
    rubrics: Emotional Intensity, Emotional Persuasiveness, Logical
    Persuasiveness, Informativeness, Lifelike.

Les notes sont renvoyees au format JSON strict. Un retry est effectue si le
parsing echoue (temperature ramenee a 0 au 2e essai).
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


SCORER_SYSTEM_PROMPT = (
    "You are an expert evaluator of empathetic conversational recommender "
    "systems. Given a dialogue history between a user and a recommender, "
    "and a candidate assistant reply that recommends a movie, you score the "
    "reply across 5 dimensions on a 0-9 integer scale. These dimensions are "
    "defined exactly as in Zhang et al. 2024 'Towards Empathetic "
    "Conversational Recommender Systems' (RecSys 2024), Section 5.3:\n\n"
    "1. Emotional Intensity (Emo Int): measures the strength of emotions "
    "conveyed to users. 0 means the reply is emotionally flat; 9 means it "
    "conveys strong, genuine emotion.\n"
    "2. Emotional Persuasiveness (Emo Pers): gauges the capacity to connect "
    "with the user emotionally to persuade them. 0 means no emotional "
    "connection; 9 means a highly emotionally persuasive recommendation.\n"
    "3. Logic Persuasiveness (Log Pers): evaluates the use of logical "
    "reasoning and coherent arguments to persuade the user. 0 means "
    "incoherent or non-sequitur; 9 means a tight, well-reasoned argument.\n"
    "4. Informativeness (Info): determines the utility of useful information "
    "provided by the system (movie themes, plot cues, cast, genre). 0 means "
    "vague; 9 means specific, useful and on-topic.\n"
    "5. Lifelikeness (Life): assesses how vivid and engaging the response "
    "is, reflecting its resemblance to natural human communication. 0 means "
    "robotic or stilted; 9 means indistinguishable from a human speaker.\n\n"
    "Output STRICT JSON only, with these exact keys in this exact order: "
    '{"Emo Int": <int 0-9>, "Emo Pers": <int 0-9>, "Log Pers": <int 0-9>, '
    '"Info": <int 0-9>, "Life": <int 0-9>}. No preamble, no explanation, '
    "no code fences."
)

SCORER_USER_TEMPLATE = (
    "Dialogue history:\n{history}\n\n"
    "Recommended movie: {item}\n\n"
    "Assistant reply to evaluate:\n{reply}\n\n"
    "Return the JSON object now."
)

METRICS = ["Emo Int", "Emo Pers", "Log Pers", "Info", "Life"]


# ---------------------------------------------------------------------------
# vLLM server lifecycle (optional helper)
# ---------------------------------------------------------------------------


def launch_vllm_server(
    model: str,
    port: int = 8000,
    quantization: Optional[str] = "awq",
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.9,
    log_path: Optional[Path] = None,
    extra_args: Optional[List[str]] = None,
) -> Optional[subprocess.Popen]:
    """Demarre `vllm serve` en background. Retourne le handle du process.

    Retourne None si `vllm` n'est pas installe (impossible d'importer).
    A killer via `kill_vllm_proc_tree(proc)` en fin de scoring pour garantir
    que les enfants (EngineCore, NCCL workers) soient tous SIGTERMs.

    On utilise `os.setsid` (via `start_new_session`) pour que le process vLLM
    demarre dans un NOUVEAU process group dont il est le leader. Cela permet
    ensuite a `os.killpg(os.getpgid(proc.pid), SIGTERM)` de propager le signal
    a TOUS les enfants -- sinon ils peuvent survivre en zombies/orphelins et
    retenir de la VRAM (observe dans les dumps live : `[python3] <defunct>
    ppid=1` apres un simple proc.terminate()).
    """
    import shutil as _sh
    if _sh.which("vllm") is None:
        print("[scorer] binaire `vllm` introuvable - `pip install vllm` requis")
        return None

    cmd = [
        "vllm", "serve", model,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]
    if quantization:
        cmd += ["--quantization", quantization]
    if extra_args:
        cmd += list(extra_args)
    log_path = Path(log_path) if log_path else Path("logs/vllm_scorer.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w")
    print(f"[scorer] vllm serve {model} -> {log_path}")
    try:
        popen_kwargs = {
            "stdout": log_f,
            "stderr": subprocess.STDOUT,
            # `start_new_session=True` == os.setsid() POSIX. Le parent et le
            # shell Python restent dans leur session d'origine.
            "start_new_session": True,
        }
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except Exception as exc:
        print(f"[scorer] echec demarrage vllm: {exc}")
        log_f.close()
        return None
    return proc


def _pgid_has_members(pgid: int) -> bool:
    """True s'il reste au moins un process VIVANT dans ce process group.

    Les zombies (STAT=Z) sont exclus : ils sont deja morts (VRAM liberee par
    le kernel), ils attendent juste d'etre reap'es par leur parent (init).
    Les compter comme "survivants" declencherait un SIGKILL inutile et
    loggerait faussement un timeout a chaque teardown.
    """
    try:
        r = subprocess.run(
            ["ps", "-e", "-o", "pid,pgid,stat"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        # FileNotFoundError herite de OSError, meme catch.
        return False
    if r.returncode != 0:
        return False
    token = f" {pgid} "
    for line in r.stdout.splitlines():
        if token not in line:
            continue
        # Dernier champ = STAT (ex: "S", "R", "Z<", "D+", etc.)
        parts = line.split()
        if len(parts) < 3:
            continue
        stat = parts[-1]
        if stat.startswith("Z"):
            continue  # zombie -> process mort, ne compte pas
        return True
    return False


def kill_vllm_proc_tree(proc: Optional[subprocess.Popen], timeout: int = 30) -> None:
    """Envoie SIGTERM (puis SIGKILL) a tout le process group de `proc`.

    Utilise `os.killpg(os.getpgid(proc.pid), SIGTERM)` pour atteindre les
    enfants crees par vLLM (EngineCore, workers NCCL). Sans ca, seul le
    wrapper principal recoit SIGTERM et les enfants deviennent orphelins
    re-adoptes par init -- observe en live sur 5090 avec des zombies
    `[python3] <defunct> ppid=1` qui pouvaient retenir de la VRAM.

    IMPORTANT : `proc.wait()` ne voit QUE le process direct de `Popen`. Si
    l'enfant immediat meurt rapidement mais laisse un petit-enfant actif
    dans le meme pgid, proc.wait returns OK alors que le pgid contient
    encore des survivants. On verifie donc avec `pgrep -g <pgid>` et on
    escalade en SIGKILL si besoin. Idempotent.
    """
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        # Process deja mort ou pgid inaccessible -> fallback sur proc direct.
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=timeout)
            except Exception:
                # TimeoutExpired et autres -> SIGKILL direct.
                try:
                    proc.kill()
                except Exception:
                    pass
        return

    # 1. SIGTERM au groupe entier (cooperation-friendly).
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return

    # Attendre que le groupe ENTIER se vide, pas juste le process direct.
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None and not _pgid_has_members(pgid):
            return
        time.sleep(0.1)

    # 2. Timeout -> SIGKILL au groupe.
    if _pgid_has_members(pgid):
        print(f"[scorer] kill_vllm_proc_tree: SIGTERM timeout {timeout}s "
              f"(pgid={pgid} non vide) -> SIGKILL")
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        # Attendre le vidage final (max 10s supplementaires).
        deadline2 = time.time() + 10
        while time.time() < deadline2:
            if not _pgid_has_members(pgid):
                break
            time.sleep(0.1)
    # Reap du process direct si pas encore fait.
    if proc.poll() is None:
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


def wait_vllm_ready(port: int = 8000, timeout: int = 600) -> bool:
    """Poll `/health` jusqu'a HTTP 200 ou timeout (defaut 10 min pour AWQ 32B)."""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    print(f"[scorer] vllm ready on :{port}")
                    return True
        except OSError:
            # URLError herite de OSError, meme catch.
            pass
        time.sleep(5)
    print(f"[scorer] timeout apres {timeout}s")
    return False


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------


_FALLBACK_RE = re.compile(r'"?([A-Z][a-zA-Z ]+?)"?\s*:\s*(\d)')


def _parse_scores(text: str) -> Optional[Dict[str, int]]:
    """Tolerant : 1) json strict ; 2) extrait `{...}` ; 3) regex fallback."""
    txt = text.strip()
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", txt, re.DOTALL)
        if not match:
            data = None
        else:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None

    if isinstance(data, dict):
        out = {k: int(data[k]) for k in METRICS if k in data and str(data[k]).isdigit()}
        if len(out) == len(METRICS):
            return out

    pairs = dict(_FALLBACK_RE.findall(txt))
    out = {k: int(pairs[k]) for k in METRICS if k in pairs}
    if len(out) == len(METRICS):
        return out
    return None


def score_one(
    client,
    model: str,
    history: str,
    item: str,
    reply: str,
    max_retries: int = 2,
) -> Optional[Dict[str, int]]:
    """Un appel chat completions + parsing robuste + retry."""
    user = SCORER_USER_TEMPLATE.format(history=history, item=item, reply=reply)
    for attempt in range(max_retries + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SCORER_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=80,
            temperature=0.0 if attempt > 0 else 0.2,
        )
        text = resp.choices[0].message.content
        parsed = _parse_scores(text)
        if parsed is not None:
            return parsed
    print(f"[scorer] echec parsing apres {max_retries + 1} essais :: {text[:200]}")
    return None


def score_system(
    client,
    model: str,
    generations: List[Dict],
    sample_size: int = 200,
    seed: int = 42,
    max_workers: int = 16,
) -> pd.DataFrame:
    """Score `sample_size` generations en parallele via ThreadPoolExecutor.

    vLLM gere nativement le continuous batching : N requetes concurrentes
    saturent le GPU et donnent un speedup de 4-8x vs. sequentiel.

    `max_workers=16` est un bon defaut sur RTX 5090 + Qwen2.5-32B-AWQ : on
    sature le scheduler sans depasser le max_model_len en memoire KV cache.
    Reduire a 8 si OOM, monter a 32 si le serveur supporte.

    Retourne un DataFrame avec `dialogue_id + item + 5 scores`. Les lignes
    dont le parsing a echoue ne sont PAS incluses (toujours cohérent).
    """
    import random
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rng = random.Random(seed)
    pool = list(generations)
    rng.shuffle(pool)
    pool = pool[:sample_size]

    def _score(row):
        return row, score_one(
            client,
            model=model,
            history=row.get("history", "") or "",
            item=row.get("item", "") or "",
            reply=row.get("response", "") or "",
        )

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_score, row) for row in pool]
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            try:
                row, scores = fut.result()
            except Exception as exc:
                print(f"[scorer] worker exception: {exc}")
                continue
            if scores is None:
                continue
            scores.update({"dialogue_id": row.get("dialogue_id"), "item": row.get("item")})
            rows.append(scores)
            if completed % 25 == 0:
                print(f"[scorer] {completed}/{len(pool)} scored ({len(rows)} valides)")
    return pd.DataFrame(rows)


def aggregate_subjective(per_system: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Moyenne par systeme -> Table 2 (colonnes: Model + 5 metriques)."""
    rows = []
    for model_name, df in per_system.items():
        if df.empty:
            continue
        mean_row = {"Model": model_name}
        for m in METRICS:
            if m in df.columns:
                mean_row[m] = float(df[m].mean())
        rows.append(mean_row)
    return pd.DataFrame(rows)

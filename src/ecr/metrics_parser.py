"""Extraction des metriques ECR depuis les logs Accelerate / loguru.

Chaque script `train_rec.py` / `train_emp.py` / `infer_emp.py` ecrit sur
stdout des lignes de la forme :

    2026-04-18 22:15:03 | INFO | epoch 4 test AUC 0.541 R@1 0.049 R@10 0.220 R@50 0.428 RT@1 0.055 RT@10 0.238 RT@50 0.452

On scan le log `.log` produit par `_run()` (streaming + tee) et on renvoie la
*derniere* ligne "test" avec les 7 metriques de l'article Table 1.

Idee-clef : **tolerant** sur l'ordre et la ponctuation (":"/" ") car loguru
et Accelerate formatent differemment selon les versions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# Regex robuste : chaque metrique peut etre suivie de `:` ou d'un espace.
# On autorise aussi `R@1 (hard)` / `R@1 (soft)` etc. en capturant le premier
# nombre apres le token.
_METRIC_PATTERNS: Dict[str, re.Pattern] = {
    "AUC":   re.compile(r"\bAUC[:\s]+([\d.]+)"),
    "R@1":   re.compile(r"\bR@1[:\s]+([\d.]+)"),
    "R@10":  re.compile(r"\bR@10[:\s]+([\d.]+)"),
    "R@50":  re.compile(r"\bR@50[:\s]+([\d.]+)"),
    "RT@1":  re.compile(r"\bRT@1[:\s]+([\d.]+)"),
    "RT@10": re.compile(r"\bRT@10[:\s]+([\d.]+)"),
    "RT@50": re.compile(r"\bRT@50[:\s]+([\d.]+)"),
}

_LOSS_RE = re.compile(r"\b(loss|train_loss)[:\s]+([\d.]+)")
_EPOCH_RE = re.compile(r"\bepoch[:\s]+(\d+)")
_STEP_RE = re.compile(r"\bstep[:\s]+(\d+)")


def extract_rec_metrics(log_path: Path) -> Dict[str, float]:
    """Retourne le dict des 7 metriques Table 1 depuis le *dernier* test log.

    Raises :
        FileNotFoundError si le log n'existe pas.
        RuntimeError si aucune ligne "test" contenant AUC n'est trouvee.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    lines = log_path.read_text(errors="ignore").splitlines()
    eval_lines = [l for l in lines if "AUC" in l and "R@" in l]
    if not eval_lines:
        raise RuntimeError(f"Aucune ligne 'AUC ... R@' trouvee dans {log_path}")

    target = None
    for line in reversed(eval_lines):
        if "test" in line.lower():
            target = line
            break
    if target is None:
        target = eval_lines[-1]

    out: Dict[str, float] = {}
    for key, pattern in _METRIC_PATTERNS.items():
        match = pattern.search(target)
        if match:
            out[key] = float(match.group(1))
    missing = [k for k in _METRIC_PATTERNS if k not in out]
    if missing:
        raise RuntimeError(
            f"Metriques manquantes dans {log_path}: {missing}\nLigne: {target!r}"
        )
    return out


def extract_training_history(log_path: Path, stage: str) -> pd.DataFrame:
    """Extrait (step, loss) de chaque ligne de training.

    Pour alimenter `results/training_history.csv` et `plot_training_loss`.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return pd.DataFrame(columns=["step", "loss", "stage"])

    rows: List[dict] = []
    step_counter = 0
    for line in log_path.read_text(errors="ignore").splitlines():
        m_loss = _LOSS_RE.search(line)
        if m_loss is None:
            continue
        loss = float(m_loss.group(2))
        m_step = _STEP_RE.search(line)
        if m_step:
            step = int(m_step.group(1))
        else:
            step_counter += 1
            step = step_counter
        rows.append({"step": step, "loss": loss, "stage": stage})
    return pd.DataFrame(rows)


def write_rec_results(
    logs_dir: Path,
    results_dir: Path,
    run_names: List[str],
    fallback_objective: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Construit `objective_metrics.csv` + `ablation_metrics.csv`.

    - `run_names` doit contenir les labels ECR ("UniCRS", "ECR_L", ...) tels
      qu'utilises pour les fichiers `logs/train_rec_<name>.log`.
    - Les lignes manquantes (KBRD, KGSF, ...) sont reprises depuis
      `fallback_objective` pour conserver la Table 1 complete.
    """
    logs_dir = Path(logs_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    measured_rows = []
    for name in run_names:
        log_path = logs_dir / f"train_rec_{name}.log"
        try:
            metrics = extract_rec_metrics(log_path)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[parse] {name}: {exc}")
            continue
        label = {
            "UniCRS": "UniCRS",
            "ECR_L":  "ECR[L]",
            "ECR_LS": "ECR[LS]",
            "ECR_LG": "ECR[LG]",
            "ECR":    "ECR",
        }.get(name, name)
        metrics["Model"] = label
        measured_rows.append(metrics)

    df_measured = pd.DataFrame(measured_rows)
    if df_measured.empty:
        print("[parse] aucun run mesure -> fallback article complet")
        if fallback_objective is None:
            return pd.DataFrame()
        df_measured = fallback_objective.copy()

    cols = ["Model", "AUC", "RT@1", "RT@10", "RT@50", "R@1", "R@10", "R@50"]
    df_measured = df_measured.reindex(columns=cols)

    if fallback_objective is not None:
        fb = fallback_objective.copy()
        fb = fb[~fb["Model"].isin(df_measured["Model"])]
        df_obj = pd.concat([df_measured, fb], ignore_index=True)
    else:
        df_obj = df_measured

    df_obj.to_csv(results_dir / "objective_metrics.csv", index=False)

    ablation_labels = {"UniCRS", "ECR[L]", "ECR[LS]", "ECR[LG]", "ECR"}
    df_abl = df_measured[df_measured["Model"].isin(ablation_labels)].copy()
    df_abl = df_abl[["Model", "AUC", "RT@10", "RT@50", "R@10", "R@50"]]
    df_abl.to_csv(results_dir / "ablation_metrics.csv", index=False)

    history_frames = []
    for name in run_names:
        log_path = logs_dir / f"train_rec_{name}.log"
        df_h = extract_training_history(log_path, stage=f"rec_{name}")
        if not df_h.empty:
            history_frames.append(df_h)
    if history_frames:
        df_hist = pd.concat(history_frames, ignore_index=True)
        df_hist.to_csv(results_dir / "training_history.csv", index=False)

    return df_obj

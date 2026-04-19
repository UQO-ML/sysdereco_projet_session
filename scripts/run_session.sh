#!/usr/bin/env bash
# Lance la session d'entrainement ECR en CLI, sans avoir le notebook ouvert.
#
# Usage (depuis la racine du projet, dans le container Docker) :
#   bash scripts/run_session.sh           # pipeline complet 24-30h
#   PHASE=smoke bash scripts/run_session.sh  # juste le smoke test GPU
#   PHASE=sweep bash scripts/run_session.sh  # uniquement Phase 2 (Table 1/3)
#   PHASE=llm   bash scripts/run_session.sh  # uniquement Phase 3 (Llama)
#   PHASE=score bash scripts/run_session.sh  # uniquement Phase 4 (Qwen scorer)
#
# La logique est volontairement la meme que la cellule 13 du notebook, sauf
# qu'ici on execute les cellules via `jupyter nbconvert --to notebook --execute`
# cellule par cellule pour avoir un log clair + possibilite de reprise.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PHASE="${PHASE:-full}"
NOTEBOOK="ecr_experiment.ipynb"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"

log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$LOG_DIR/session.log"; }

run_py() {
    local desc="$1"; shift
    local script="$*"
    log "=== $desc ==="
    python3 - <<PYEOF 2>&1 | tee -a "$LOG_DIR/session.log"
import json, sys, types
from pathlib import Path
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))

# Execute les cellules du notebook en ordre, dans un meme namespace global.
ns = {}
with open("$NOTEBOOK") as f:
    nb = json.load(f)
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    exec(compile(src, f"<{cell.get('id','cell')}>", "exec"), ns)
$script
PYEOF
}

phase_smoke() {
    run_py "Smoke test materiel" ''
}

phase_prep() {
    run_py "Environnement + donnees (clone + patch + archives + ReDial prep)" '
cfg = ns["get_config"]()
ns["clone_ecr_repo"](cfg)
ns["patch_ecr_compat"](cfg)
ns["download_external_assets"](cfg)
ns["install_base_models"](cfg)
ns["install_pretrained_ckpts"](cfg)
ns["prepare_redial_data"](cfg)
'
}

phase_sweep() {
    run_py "Phase 2 - sweep Table 1/3 (5 variantes train_rec)" '
cfg = ns["get_config"]()
ns["clone_ecr_repo"](cfg)
ns["patch_ecr_compat"](cfg)
ns["install_pretrained_ckpts"](cfg)
ns["run_recommendation_sweep"](cfg)
ns["build_results_from_logs"](cfg)
'
}

phase_llm() {
    run_py "Phase 3 - generation ECR[DialoGPT] + Llama zero-shot + ECR[Llama 2-Chat]" '
cfg = ns["get_config"]()
ns["clone_ecr_repo"](cfg)
ns["patch_ecr_compat"](cfg)
ns["install_pretrained_ckpts"](cfg)
ns["run_response_generation_training"](cfg)
ns["run_llama_zero_shot"](cfg)
ns["run_llama_lora_train"](cfg)
ns["run_llama_lora_generate"](cfg)
'
}

phase_score() {
    run_py "Phase 4 - Qwen2.5-32B-AWQ scorer + Cohen kappa" '
cfg = ns["get_config"]()
ns["run_llm_scorer"](cfg)
df_obj, df_subj_llm, df_subj_human, df_ablation = ns["load_results_data"](cfg)
ns["compute_scorer_kappa"](cfg, df_subj_llm, df_subj_human)
'
}

phase_full() {
    phase_smoke
    phase_prep
    phase_sweep
    phase_llm
    phase_score
    log "=== Session complete ==="
}

case "$PHASE" in
    smoke) phase_smoke ;;
    prep)  phase_prep  ;;
    sweep) phase_sweep ;;
    llm)   phase_llm   ;;
    score) phase_score ;;
    full)  phase_full  ;;
    *)     echo "PHASE inconnue: $PHASE (smoke|prep|sweep|llm|score|full)"; exit 2 ;;
esac

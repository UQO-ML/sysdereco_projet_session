#!/usr/bin/env bash
# Lance la session d'entrainement ECR en CLI, sans avoir le notebook ouvert.
#
# Usage (depuis la racine du projet, dans le container Docker) :
#   bash scripts/run_session.sh           # pipeline complet
#   PHASE=smoke bash scripts/run_session.sh   # juste l'intro environnement
#   PHASE=prep  bash scripts/run_session.sh   # clone + assets + ReDial
#   PHASE=pre   bash scripts/run_session.sh   # train_pre.py
#   PHASE=rec   bash scripts/run_session.sh   # train_rec.py + export rec.json
#   PHASE=gen   bash scripts/run_session.sh   # merge_rec + train_emp + infer_emp
#   PHASE=export bash scripts/run_session.sh  # export_run_metrics + diagnose_dataset
#   PHASE=llm   bash scripts/run_session.sh   # Llama zero-shot + ECR[Llama 2-Chat]
#   PHASE=score bash scripts/run_session.sh   # Scorer Qwen + kappa
#
# La logique reste alignee avec le notebook (les fonctions Python sont
# exportees par l'execution de toutes les cellules code en ordre).

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
import json, sys
from pathlib import Path
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))

# Execute toutes les cellules code du notebook en ordre, dans un namespace partage.
ns = {}
with open("$NOTEBOOK") as f:
    nb = json.load(f)
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    try:
        exec(compile(src, f"<cell:{cell.get('id','?')}>", "exec"), ns)
    except Exception as exc:
        # On continue l'execution des cellules : certaines (ex: visus) peuvent
        # dependre de variables definies plus loin. Les appels effectifs
        # ci-dessous reverifient la presence des fonctions necessaires.
        print(f"[warn] cell load error (ignored to keep defs): {exc}")
# Commandes specifiques de la phase :
cfg = ns["get_config"]()
$script
PYEOF
}

phase_smoke()  { run_py "Smoke test materiel" ''; }
phase_prep() {
    run_py "Environnement + assets + ReDial" '
ns["clone_ecr_repo"](cfg)
ns["patch_ecr_compat"](cfg)
ns["download_external_assets"](cfg)
ns["install_base_models"](cfg)
ns["install_pretrained_ckpts"](cfg)
ok = ns["prepare_redial_data"](cfg)
ns["diagnose_dataset"](cfg)
'
}
phase_pre() {
    run_py "train_pre.py" '
ns["clone_ecr_repo"](cfg); ns["patch_ecr_compat"](cfg)
assert ns["run_emotional_semantic_fusion"](cfg) is not False, "train_pre failed"
ns["export_run_metrics"](cfg)
'
}
phase_rec() {
    run_py "train_rec.py + export rec.json" '
ns["clone_ecr_repo"](cfg); ns["patch_ecr_compat"](cfg)
assert ns["run_recommendation_training"](cfg) is not False, "train_rec failed"
ns["export_run_metrics"](cfg)
'
}
phase_gen() {
    run_py "merge_rec + train_emp + infer_emp" '
ns["clone_ecr_repo"](cfg); ns["patch_ecr_compat"](cfg)
assert ns["run_response_generation_training"](cfg) is not False, "generation failed"
ns["export_run_metrics"](cfg)
'
}
phase_export() {
    run_py "Export metriques (results/*.csv cumulatif)" '
ns["diagnose_dataset"](cfg)
ns["export_run_metrics"](cfg)
'
}
phase_llm() {
    run_py "Llama zero-shot + ECR[Llama 2-Chat]" '
cfg["run_llama_zero_shot"] = True
cfg["run_llama_lora"] = True
ns["run_llama_zero_shot"](cfg)
ns["run_llama_lora_train"](cfg)
ns["run_llama_lora_generate"](cfg)
'
}
phase_score() {
    run_py "Scorer Qwen + kappa" '
cfg["run_llm_scorer"] = True
cfg["run_kappa"] = True
ns["_ensure_dialogpt_generations"](cfg) if "_ensure_dialogpt_generations" in ns else None
ns["run_llm_scorer"](cfg)
df_obj, df_subj_llm, df_subj_human, df_ablation = ns["load_results_data"](cfg)
ns["compute_scorer_kappa"](cfg, df_subj_llm, df_subj_human)
'
}
phase_full() {
    phase_smoke
    phase_prep
    phase_pre
    phase_rec
    phase_gen
    phase_export
    phase_llm
    phase_score
    log "=== Session complete ==="
}

case "$PHASE" in
    smoke)  phase_smoke  ;;
    prep)   phase_prep   ;;
    pre)    phase_pre    ;;
    rec)    phase_rec    ;;
    gen)    phase_gen    ;;
    export) phase_export ;;
    llm)    phase_llm    ;;
    score)  phase_score  ;;
    full)   phase_full   ;;
    *)      echo "PHASE inconnue: $PHASE (smoke|prep|pre|rec|gen|export|llm|score|full)"; exit 2 ;;
esac

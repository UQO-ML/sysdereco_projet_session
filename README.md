# sysdereco_projet_session

Reproduction **quasi-integrale** sur une seule RTX 5090 32 GB de l'article :

> **Towards Empathetic Conversational Recommender Systems** — *RecSys 2024*  
> Xiaoyu Zhang, Ruobing Xie, Yougang Lyu, Xin Xin, Pengjie Ren, Mingfei Liang, Bo Zhang, Zhanhui Kang, Maarten de Rijke, Zhaochun Ren  
> [doi:10.1145/3640457.3688133](https://doi.org/10.1145/3640457.3688133)

Le code de reference est celui du depot officiel [zxd-octopus/ECR](https://github.com/zxd-octopus/ECR). Le notebook clone, patche (21 patches idempotents pour rendre le code compatible avec la stack moderne `transformers >= 4.40` / `pyg >= 2.5` / `accelerate >= 0.28` / `torch >= 2`), entraine, genere et evalue — le tout en **~18-22 h** avec les optimisations activees.

## Ce qui est *reellement mesure* vs *fallback article*

| Table de l'article | Dans ce projet |
|---|---|
| **Table 1** (Recommandation RQ1) | `UniCRS` + `ECR` **mesures**, KBRD/KGSF/RevCore/UCCR en **fallback article** |
| **Table 2** (Generation RQ2) | `ECR[DialoGPT]` + `Llama 2-7B-Chat` + `ECR[Llama 2-Chat]` **mesures** ; scorer GPT-4-turbo remplace par **Qwen2.5-32B-Instruct-AWQ** local (Cohen κ rank-level reporte) ; `GPT-3.5-*` en fallback article (API payante) |
| **Table 3** (Ablation RQ3) | **5/5 variantes mesurees** (UniCRS, ECR[L], ECR[LS], ECR[LG], ECR) avec seed=8 fixe |

Les lignes non reproduites sont clairement marquees dans la cellule markdown "Limitations methodologiques" du notebook.

## Contenu

- **`ecr_experiment.ipynb`** — notebook unique a la racine (35 cellules), structure comme [tag_reco_experiment.ipynb](https://github.com/WebSemantic-Projet-n-1/projet_session/blob/main/tag_reco_experiment.ipynb) :
  - une fonction par cellule, une seule cellule d'import de donnees ;
  - chaque section reliee a la section correspondante de l'article (4.1, 4.2, 4.3, 5.3-5.6) ;
  - derniere cellule = resume + 12 visualisations + Cohen κ.
- **`src/ecr/`** — helpers Python importables depuis le notebook :
  - `metrics_parser.py` : parse les logs `train_rec_*.log` -> `results/objective_metrics.csv` + `ablation_metrics.csv` + `training_history.csv` ;
  - `llama_runner.py` : generation Llama 2-7B-Chat (zero-shot + ECR[Llama 2-Chat]) via HF bf16 ou vLLM ;
  - `llama_lora.py` : fine-tuning LoRA fidele a la Section 5.4 (r=16, α=32, q/k/v/o_proj, 15 epochs, eff batch 16, lr=5e-5) ;
  - `scorer.py` : serveur vLLM Qwen2.5-32B-AWQ + prompt Appendix E (5 rubriques 0-9) + scoring concurrent (ThreadPoolExecutor).
- **`src/viz/plots.py`** — 12 visualisations centralisees (distribution feedback, emotions, couverture reviews, metriques objectives/subjectives, ablation, LLM vs Human, training loss, sweep).
- **`scripts/run_session.sh`** — runbook CLI par phase (`smoke` / `prep` / `sweep` / `llm` / `score` / `full`) pour lancer overnight sans notebook ouvert.
- **`docker-compose.override.yml`** — service `scorer` optionnel (profil `scorer`) pour lancer vLLM dans un container separe.
- `ECR/` (cree au premier lancement) — clone automatique de [zxd-octopus/ECR](https://github.com/zxd-octopus/ECR).
- `data/emo_data/` et `data/ckpt/` — archives fournies par les auteurs, telechargees via `gdown` :
  - [`emo_data.zip` (~111 MB)](https://drive.google.com/file/d/1fb9kDo8uSRLlwc5c4nUw8DZHR5XOY_l_/view) : annotations GPT-3.5, reviews IMDb filtrees, entites DBpedia, `llama_train.json`, `llama_test.json` ;
  - [`ckpt.zip` (~679 MB)](https://drive.google.com/file/d/1uBtcqbQByVrrJ1hEwk2dvsAOxuvEgE19/view) : poids pre-entraines (GPT-2 emotion classifier + ECR[DialoGPT] + ECR[Llama 2-Chat]).
- `results/` — CSV mesures (ou fallback article si absents) + `generations/` (jsonl des reponses generees) + `scorer_kappa.csv`.
- `logs/` — tee des runs `accelerate launch` (streaming ligne a ligne via `_run()`).

## Arborescence

```text
.
├── ecr_experiment.ipynb
├── src/
│   ├── ecr/
│   │   ├── __init__.py
│   │   ├── metrics_parser.py
│   │   ├── llama_runner.py
│   │   ├── llama_lora.py
│   │   └── scorer.py
│   └── viz/
│       └── plots.py
├── scripts/
│   └── run_session.sh
├── ECR/                     # clone auto (depot officiel ECR)
├── data/
│   ├── emo_data/            # archive Google Drive dezippee
│   └── ckpt/                # archive Google Drive dezippee
├── results/                 # CSV mesures (ou fallback article sinon)
│   ├── generations/         # *.jsonl (ecr_dialogpt, llama2_zero_shot, ecr_llama_chat)
│   ├── objective_metrics.csv      (Table 1)
│   ├── subjective_metrics_llm.csv (Table 2 - Qwen scorer)
│   ├── subjective_metrics_human.csv (Table 2 - fallback humains article)
│   ├── ablation_metrics.csv       (Table 3)
│   ├── training_history.csv       (loss par stage)
│   └── scorer_kappa.csv           (Cohen κ scorer vs humains article)
├── logs/                    # tee *.log des subprocess
├── 3640457.3688133.pdf      # article
├── Dockerfile
├── docker-compose.yml
├── docker-compose.override.yml   # profil `scorer` (vllm dedie)
├── requirements.txt
└── README.md
```

## Stack

Alignement article (Section 5.4) + compatibilite stack moderne :

- **Deep learning** : PyTorch 2.x (image `nvcr.io/nvidia/pytorch:26.03-py3`, CUDA 12.8, support Blackwell sm_120)
- **LLM tooling** : `transformers>=4.40,<5`, `accelerate>=0.28`, `peft>=0.10` (LoRA Llama 2-Chat), `datasets`
- **Graph** : `torch-geometric>=2.5` (RGCN Section 4.2)
- **Quantization** : `bitsandbytes` (QLoRA 4-bit optionnel), AWQ via vLLM
- **Inference/scoring rapide** : `vllm>=0.6` + `openai>=1.30` (client local)
- **Attention** : `flash-attn>=2.8` (optionnel, fallback `sdpa` si absent)
- **Metriques** : `pandas`, `numpy`, `scikit-learn`
- **Visualisation** : `matplotlib`, `seaborn`

## Lancement Docker (recommande)

```bash
# 1. Creer .env avec ton HF_TOKEN (Llama 2 est gated, licence gratuite sur HF)
echo "HF_TOKEN=hf_xxx" > .env

# 2. Build + up
docker compose up --build -d

# 3. Notebook sur http://localhost:8890 (token: sysdereco)
```

Scorer vLLM dans un container dedie (profil optionnel) :

```bash
docker compose --profile scorer up -d scorer
```

## Lancement CLI (overnight, sans notebook ouvert)

```bash
# Pipeline complet (24-30h sans optim, ~18-22h avec optim par defaut)
bash scripts/run_session.sh

# Phase par phase (reprise apres echec)
PHASE=smoke bash scripts/run_session.sh   # 1min  - valide sm_120
PHASE=prep  bash scripts/run_session.sh   # 30min - clone + patch + archives + ReDial
PHASE=sweep bash scripts/run_session.sh   # ~10h  - 5 variantes train_rec (Table 1/3)
PHASE=llm   bash scripts/run_session.sh   # ~8h   - DialoGPT + Llama zero-shot + LoRA
PHASE=score bash scripts/run_session.sh   # ~30min-2h - Qwen scorer (avec concurrence)
```

## Patches de compatibilite (`patch_ecr_compat`)

Le code ECR a ete ecrit pour `transformers 4.15 / pyg 2.0.1 / accelerate 0.8 / torch 1.8`. **21 patches idempotents** sont appliques au clone pour rendre le code compatible avec la stack moderne :

| Patch | Cible | Raison |
|---|---|---|
| 1 | `transformers.AdamW` -> `torch.optim.AdamW` | retire en 4.30+ |
| 2, 3 | `model_gpt2.py` imports | `find_pruneable_heads_and_indices` deplace + `model_parallel_utils` retire |
| 4 | `SparseTensor` -> tensor standard | pyg 2.5+ requiert `torch-sparse` sinon |
| 5 | `RGCNConv(..., edge_index, edge_type)` | forme canonique 3-args |
| 6 | `Accelerator(fp16=...)` retire | `--mixed_precision bf16` via `accelerate launch` |
| 7 | `torch.set_deterministic` + `CUDA_LAUNCH_BLOCKING` | deprecie / 5-10× plus lent sur GPU moderne |
| 8 | `accelerator.use_fp16` retire | fallback `getattr(..., False)` |
| 9 | `.cuda()` hardcode dans `dataset_dbpedia.py` | `.to(device)` pour tolerer CPU transitoire |
| 10 | `print("here")` debug auteur | 5 occurrences nettoyees |
| 11 | `evaluate_rec.py` : `mkdir save/redial_rec/` | avant ecriture `rec.json` |
| 12 | `transformers.file_utils` -> `transformers.utils` | deprecie 4.25, retire 5.0 |
| 13 | `PromptGPT2forCRS` + `GenerationMixin` | `.generate()` post transformers 4.50 |
| 14 | `CUDA_VISIBLE_DEVICES=3` retire | forcait GPU 3 (multi-GPU auteur) |
| 15 | `os.makedirs("log")` avant `logger.add(...)` | loguru crash sinon |
| 16-18 | chemins de sortie stables | `data/saved/{pre,rec,emp}` sans timestamp |
| 19 | `PromptGPT2forCRS.from_pretrained(args.model, local_files_only=...)` | `HFValidationError` sur dossier local |
| 20 | `retain_graph=True` retire | fuite memoire inutile |
| 21 | 7 assertions post-patch | detection regression regex silencieuse |
| **22** | `DataLoader(num_workers=args.num_workers, pin_memory=True)` | **optim** : workers DataLoader dans `train_pre.py`/`train_rec.py` (amont parse `--num_workers` mais ne le passe pas) |
| **23** | `torch.compile(prompt_encoder, mode='reduce-overhead')` | **optim** : 1.2-1.5× wall-time apres warmup (try/except fallback eager) |

## Optimisations (actives par defaut sauf mention)

| # | Optimisation | Gain | Risque qualite |
|---|---|---|---|
| 1 | **DataLoader `num_workers=8` + `pin_memory`** (Patch 22 + CLI) | 20-30% sur phases 2/3 | zero |
| 2 | **Scoring concurrent** (`ThreadPoolExecutor(max_workers=16)` vers vLLM) | 4-8× sur Phase 4 | zero |
| 3 | **`torch.compile(prompt_encoder)`** (Patch 23) | 1.2-1.5× sur Phase 2 | zero |
| 4 | **Flash-Attention 2** (Llama inference + LoRA training, fallback `sdpa`) | 2-3× sur attention Llama | zero |
| 5 | **bf16 mixed precision** (`mixed_precision: "bf16"`) | 1.5-2× wall-time | negligeable |
| 6 | **batch_scale=2.0** (double batch, divise accum) | 20-40% sur entrainements | zero (eff batch preservee) |
| **opt-in** | **QLoRA 4-bit** (`cfg["lora_use_4bit"] = True`) | 2× sur LoRA + VRAM /4 | < 0.7% (negligeable sur 2459 samples) |
| **opt-in** | **vLLM Llama auto-start** (`cfg["llama_vllm_autostart"] = True`) | 10× sur generation Llama | zero |

Total estime : **24-30h** -> **18-22h** avec defauts ; **14-17h** avec les 2 opt-in.

## Donnees externes

Les archives `emo_data.zip` (~111 MB) et `ckpt.zip` (~679 MB) sont partagees par les auteurs via Google Drive. Le notebook utilise [`gdown`](https://github.com/wkentaro/gdown) qui gere automatiquement l'interstitiel "virus scan warning" de Google Drive pour tout fichier > 100 MB.

En cas d'echec reseau / quota Google Drive :

1. deposer manuellement `emo_data.zip` dans `data/emo_data.zip` ;
2. deposer manuellement `ckpt.zip` dans `data/ckpt.zip` ;
3. relancer la cellule d'import de donnees — elle decompresse vers `data/emo_data/` et `data/ckpt/`.

**Llama 2-7B-Chat** (modele gated sur HuggingFace) necessite `HF_TOKEN` dans `.env` et l'acceptation de la licence (gratuit, ~5 min de validation auto sur [la page HF](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)).

## Flags d'execution (`get_config()`)

| Flag | Defaut | Effet |
|---|---|---|
| `dry_run` | `False` | `True` = skip clone/archives/trainings/inference, seulement EDA + tables article |
| `use_pretrained_ckpt` | `False` | `True` = copie `data/ckpt/{pre,rec,emp}` dans `src_emo/data/saved/` et saute les 3 `train_*.py` (gain > 20h) |
| `run_rec_sweep` | `True` | lance les 5 variantes Table 1/3 (sinon fallback article) |
| `run_llama_zero_shot` | `True` | genere `llama2_zero_shot.jsonl` pour Table 2 |
| `run_llama_lora` | `True` | fine-tune LoRA + genere `ecr_llama_chat.jsonl` |
| `run_llm_scorer` | `True` | scorer Qwen2.5-32B-AWQ sur 200 exemples × 5 rubriques |
| `run_kappa` | `True` | calcule Cohen κ rank-level scorer vs humains article |
| `batch_scale` | `2.0` | multiplicateur `per_device_batch`, divise `grad_accum` d'autant (eff batch preservee) |
| `mixed_precision` | `"bf16"` | `"bf16"` recommande Blackwell, `"fp16"` sinon, `"no"` pour fidelite article stricte |
| `dataloader_num_workers` | `8` | Patch 22 : workers DataLoader (9950X3D = 16 coeurs) |
| `use_torch_compile` | `True` | Patch 23 : `torch.compile(prompt_encoder)` |
| `use_flash_attn_2` | `True` | Llama : `attn_implementation="flash_attention_2"` (fallback `sdpa`) |
| `scorer_concurrency` | `16` | ThreadPoolExecutor workers vers vLLM |
| `scorer_sample_size` | `200` | aligne sur panel humain article (1000 pour GPT-4-turbo dans l'article) |
| `llama_vllm_autostart` | `False` | `True` = lance un serveur vLLM Llama avant generation (besoin ~15 GB VRAM) |
| `lora_use_4bit` | `False` | `True` = QLoRA nf4 (2× vitesse, VRAM /4) |

## Lancement local (venv, sans Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name sysdereco-venv --display-name "Python (sysdereco-venv)"
jupyter lab
```

Note : `flash-attn` requiert CUDA au build. Si l'install echoue, commenter la ligne dans `requirements.txt` — le code bascule automatiquement sur `sdpa` (PyTorch natif, deja 1.3-1.5× plus rapide que l'impl eager).

## Notes

- Le seed `seed_rec_runs=8` est fixe pour les 5 variantes Table 3 (comparabilite).
- Les sorties `accelerate launch` sont streamees en temps reel via `_run()` + tee dans `logs/<desc>.log`.
- `patch_ecr_compat` est **idempotent** : re-application sans effet. Utile apres `cd ECR && git checkout -- src_emo/` + re-run.
- Les graphiques sont centralises dans `src/viz/plots.py` pour garder le notebook lisible.
- Le scorer local Qwen2.5-32B-AWQ remplace GPT-4-turbo de l'Appendix E. Le Cohen κ rank-level vs humains article (publie : 0.62) permet de documenter la validite du proxy. Kappa ≥ 0.5 = substitut acceptable ; entre 0.3 et 0.5 = limite a documenter ; < 0.3 = scorer a questionner.

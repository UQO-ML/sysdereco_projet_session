# sysdereco_projet_session

Notebook d'experience et scripts de visualisation inspires de l'article :

> **Towards Empathetic Conversational Recommender Systems** — *RecSys 2024*  
> Xiaoyu Zhang, Ruobing Xie, Yougang Lyu, Xin Xin, Pengjie Ren, Mingfei Liang, Bo Zhang, Zhanhui Kang, Maarten de Rijke, Zhaochun Ren  
> [doi:10.1145/3640457.3688133](https://doi.org/10.1145/3640457.3688133)

Le code de reference est celui du depot officiel [zxd-octopus/ECR](https://github.com/zxd-octopus/ECR). Le notebook reprend fidelement les etapes du README (`train_pre.py`, `train_rec.py`, `train_emp.py`, `infer_emp.py`) avec un drapeau `DRY_RUN` pour rester executable hors GPU 24 GB.

## Contenu

- `ecr_experiment.ipynb` — **notebook unique a la racine** du projet, structure comme [tag_reco_experiment.ipynb](https://github.com/WebSemantic-Projet-n-1/projet_session/blob/main/tag_reco_experiment.ipynb) :
  - une fonction par cellule ;
  - **une seule cellule d'import de donnees** ;
  - derniere cellule = compilation + visualisations comparatives ;
  - chaque section est introduite par une cellule markdown qui relie le code a la section correspondante de l'article (4.1, 4.2, 4.3, 5.x).
- `src/viz/plots.py` — toutes les visualisations (distribution de feedback, labels d'emotion, couverture de reviews, metriques objectives/subjectives, ablation, LLM vs Human, training loss, sweep d'hyperparametres).
- `ECR/` (cree au premier lancement) — clone automatique de [`zxd-octopus/ECR`](https://github.com/zxd-octopus/ECR).
- `data/emo_data/` et `data/ckpt/` (optionnels) — archives fournies par les auteurs (telechargees via `gdown` depuis Google Drive) :
  - [`emo_data.zip` (~111 MB)](https://drive.google.com/file/d/1fb9kDo8uSRLlwc5c4nUw8DZHR5XOY_l_/view) ;
  - [`ckpt.zip` (~679 MB)](https://drive.google.com/file/d/1uBtcqbQByVrrJ1hEwk2dvsAOxuvEgE19/view).
- `results/` (optionnel) — CSV de metriques reelles. Sinon le notebook retombe sur les tables de l'article.

## Arborescence

```text
.
├── ecr_experiment.ipynb
├── src/
│   └── viz/
│       └── plots.py
├── ECR/                 # clone auto (depot officiel)
├── data/
│   ├── emo_data/        # archive a deposer manuellement si Proton Drive bloque curl
│   └── ckpt/            # idem
├── results/             # CSV optionnels : objective_metrics.csv, subjective_metrics_*.csv, ablation_metrics.csv, training_history.csv
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Stack (alignee sur l'article)

- Deep learning : `PyTorch`
- LLM tooling : `transformers`, `datasets`, `accelerate`, `peft` (LoRA pour Llama 2-Chat)
- Donnees & metriques : `pandas`, `numpy`, `scikit-learn`
- Visualisation : `matplotlib`, `seaborn`
- Notebook : `jupyterlab`
- Image de base : `nvcr.io/nvidia/pytorch`

## Lancement Docker (recommande)

```bash
docker compose up --build
```

- URL : `http://localhost:8890`
- Token : `sysdereco`
- Ouvrir `ecr_experiment.ipynb` (a la racine du workspace).

## Lancement local (venv)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name sysdereco-venv --display-name "Python (sysdereco-venv)"
jupyter lab
```

## Donnees externes

Les archives `emo_data.zip` (~111 MB) et `ckpt.zip` (~679 MB) sont partagees par les auteurs via Google Drive. Le notebook utilise [`gdown`](https://github.com/wkentaro/gdown) qui gere automatiquement l'interstitiel "virus scan warning" servi par Google Drive pour tout fichier > 100 MB (sans quoi `curl` recuperait la page HTML au lieu du binaire).

En cas d'echec reseau / quota Google Drive :

1. deposer manuellement `emo_data.zip` dans `data/emo_data.zip` ;
2. deposer manuellement `ckpt.zip` dans `data/ckpt.zip` ;
3. relancer la cellule d'import de donnees — elle decompresse vers `data/emo_data/` et `data/ckpt/`.

## Fichiers de resultats optionnels

Si presents, `ecr_experiment.ipynb` lit :

- `results/objective_metrics.csv` (Table 1)
- `results/subjective_metrics_llm.csv` (Table 2 — LLM scorer)
- `results/subjective_metrics_human.csv` (Table 2 — humains)
- `results/ablation_metrics.csv` (Table 3)
- `results/training_history.csv` (courbes de loss)

Sinon le notebook utilise automatiquement les valeurs publiees dans l'article pour rester immediatement executable.

## Notes

- Le drapeau `DRY_RUN` dans `get_config()` evite les trois entrainements GPU (> 6 h sur carte 24 GB).
- Les graphiques (`plot_*`) sont centralises dans `src/viz/plots.py` pour garder le notebook lisible.

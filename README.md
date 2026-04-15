# sysdereco_projet_session

Experiment notebook and visualization scripts inspired by:
**Towards Empathetic Conversational Recommender Systems (RecSys 2024)**.

## What is included

- `notebooks/ecr_experiment.ipynb`
  - Simple notebook structure
  - Functions split by cell
  - One dedicated data import cell
  - Final cell compiles data and plots all comparisons
- `src/viz/plots.py`
  - All plotting logic moved out of the notebook
  - Reusable functions for objective and subjective metrics

## Project layout

```text
.
├── notebooks/
│   └── ecr_experiment.ipynb
├── src/
│   └── viz/
│       └── plots.py
├── results/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Stack (aligned with the paper)

- Deep learning: `PyTorch`
- LLM tooling: `transformers`, `datasets`, `accelerate`, `peft` (LoRA)
- Data & metrics: `pandas`, `numpy`, `scikit-learn`
- Visualization: `matplotlib`, `seaborn`
- Notebook runtime: `jupyterlab`
- Container base: NVIDIA TensorFlow container (`nvcr.io/nvidia/tensorflow`)

## Run with Docker (recommended first)

### 1) Build and start

```bash
docker compose up --build
```

### 2) Open JupyterLab

- URL: `http://localhost:8890`
- Token: `sysdereco`

### 3) Open notebook

- `notebooks/ecr_experiment.ipynb`

## Run local with venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name sysdereco-venv --display-name "Python (sysdereco-venv)"
jupyter lab
```

Open `notebooks/ecr_experiment.ipynb` and select kernel `Python (sysdereco-venv)`.

## Results data files (optional)

If present, the notebook reads:

- `results/objective_metrics.csv`
- `results/subjective_metrics_llm.csv`
- `results/subjective_metrics_human.csv`

If files are missing, the notebook uses fallback values from the paper tables so it can run immediately.

## Notes

- The notebook is intentionally simple and educational.
- Visualizations are centralized in `src/viz/plots.py` for cleaner notebook cells and easier reuse.
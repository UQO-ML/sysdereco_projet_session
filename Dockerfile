# syntax=docker/dockerfile:1.6
#
# Base : image NVIDIA PyTorch (fournit CUDA + torch + torchvision + apex + TensorRT).
# L'article ECR exige CUDA 11 / torch 1.8.1 d'origine, mais on cible ici la stack
# moderne `nvcr.io/nvidia/pytorch:26.03-py3` + pyg >= 2.5 pour garder les memes
# comportements (RGCN, accelerate, LoRA) sur une image maintenue.
FROM nvcr.io/nvidia/pytorch:26.03-py3

ARG JUPYTER_PORT=8890
ARG HF_HOME=/workspace/.cache/huggingface

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    JUPYTER_PORT=${JUPYTER_PORT} \
    MPLBACKEND=Agg \
    # Caches centralises (persistables via volumes docker-compose)
    # NB: depuis transformers 5.x, seul HF_HOME suffit (TRANSFORMERS_CACHE et
    # HF_DATASETS_CACHE sont derives automatiquement et emettent un warning).
    HF_HOME=${HF_HOME} \
    HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub \
    TORCH_HOME=/workspace/.cache/torch \
    NLTK_DATA=/usr/local/share/nltk_data \
    # Accelerate : desactive l'assistant interactif dans les containers
    ACCELERATE_DISABLE_RICH=1 \
    # wandb : offline par defaut (les scripts ECR `import wandb` sans y loguer)
    WANDB_MODE=disabled \
    WANDB_DISABLED=true

WORKDIR /workspace

# Dependances systeme : git+lfs (modeles HF), outils de build (bitsandbytes, pyg),
# utilitaires requis par le notebook (unzip pour emo_data.zip / ckpt.zip).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs \
        curl wget ca-certificates \
        build-essential pkg-config \
        unzip zip \
        openssh-client \
        tzdata locales \
    && git lfs install --system \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Stack Python : HF + LoRA + PyG + outils notebook.
# L'image NGC embarque deja `pip 26.x` + tout le socle scientifique (numpy,
# pandas, torch, transformers en partie). On evite de re-upgrader ces paquets
# pour contourner `uninstall-no-record-file` sur les paquets installes via apt
# (wheel/PyYAML/pygments) et pour ne pas casser la contrainte dali/packaging.
# -> requirements.txt ne liste que les paquets *manquants* pour ECR.
COPY requirements.txt /workspace/requirements.txt

# `--no-build-isolation` est indispensable pour flash-attn : son `setup.py`
# fait `import torch` a l'etape de build, mais un build isole ne voit PAS le
# torch global de l'image NGC (il cree un venv temporaire vide). Avec
# `--no-build-isolation`, pip utilise le site-packages existant -> torch visible,
# la compilation reussit. Accessoirement ca accelere le build (pas de reinstall
# redondant des build-deps). L'ancien `|| true` masquait silencieusement cet
# echec ; tous les paquets de requirements.txt n'etaient alors PAS installes.
RUN pip install --no-build-isolation -r /workspace/requirements.txt

# --- Gel des versions torch / torchvision / torchaudio / flash-attn / vllm ---
#
# Probleme : vllm 0.19.x declare des pins stricts `torchaudio==2.10.0` +
# `torchvision==0.25.0` qui, transitivement, forcent `torch==2.10.0`. Si on
# part de la build torch 2.11.0a0 fournie par l'image NGC, le premier
# `pip install` avec vllm deprend/remplace torch (dezipagge ~1 Go de wheels
# cuda). Une fois torch remplace, ca reste ; mais tout `pip install <paquet>`
# ulterieur dans le conteneur peut a nouveau tenter de re-jouer la resolution
# de dependances si l'utilisateur n'utilise pas `--no-build-isolation`.
#
# Solution : on fige les versions observees apres install dans le fichier
# pointe par `PIP_CONSTRAINT` (convention NGC : `/etc/pip/constraint.txt`).
# Tout futur `pip install` dans le conteneur refusera alors d'upgrader ou
# downgrader ces paquets. Si un utilisateur ajoute un paquet dont la dep
# entre en conflit, pip le lui dira explicitement au lieu de reinstaller
# silencieusement ~2-3 Go de torch+cuda wheels.
RUN python - <<'PY' >> /etc/pip/constraint.txt
import importlib.metadata as _md
_PINS = (
    "torch", "torchvision", "torchaudio", "triton",
    "flash-attn", "vllm",
    "torch-geometric",
    "transformers", "accelerate", "peft", "bitsandbytes",
    "numpy", "numba", "protobuf",
)
print("# --- Auto-generated pins (Dockerfile) : empeche pip de reinstaller ---")
for _p in _PINS:
    try:
        print(f"{_p}=={_md.version(_p)}")
    except _md.PackageNotFoundError:
        pass
PY
RUN echo "[pip-constraint] contenu final de /etc/pip/constraint.txt :" \
 && cat /etc/pip/constraint.txt

# Corpus NLTK necessaires aux scripts `data/redial/process*.py` + generation
RUN mkdir -p "${NLTK_DATA}" && \
    python -m nltk.downloader -d "${NLTK_DATA}" \
        punkt punkt_tab stopwords wordnet averaged_perceptron_tagger omw-1.4

# Code du projet (remplace au runtime par le bind-mount de docker-compose)
# COPY . /workspace

# Pre-cree les dossiers que le notebook attend (l'archive `emo_data` / `ckpt`
# et le clone `ECR/` sont fournis au runtime).
RUN mkdir -p /workspace/data /workspace/results \
             "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" \
             "${TORCH_HOME}"

EXPOSE ${JUPYTER_PORT}

# Healthcheck : s'assure que JupyterLab repond sur /api
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsSL "http://localhost:${JUPYTER_PORT}/api" >/dev/null || exit 1

# JupyterLab par defaut - docker-compose surcharge pour ajouter le token/ip/port
CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=${JUPYTER_PORT} --no-browser --allow-root"]

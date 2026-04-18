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
RUN pip install -r /workspace/requirements.txt

# Corpus NLTK necessaires aux scripts `data/redial/process*.py` + generation
RUN mkdir -p "${NLTK_DATA}" && \
    python -m nltk.downloader -d "${NLTK_DATA}" \
        punkt punkt_tab stopwords wordnet averaged_perceptron_tagger omw-1.4

# Code du projet (remplace au runtime par le bind-mount de docker-compose)
COPY . /workspace

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

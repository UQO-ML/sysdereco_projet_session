FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Stack article: PyTorch + HF + LoRA + notebooks + plotting
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements.txt

COPY . /workspace

EXPOSE 8890

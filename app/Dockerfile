FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libsndfile1 \
    libgl1 \
    libsm6 \
    libxext6 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel packaging ninja

RUN pip install torch==2.12.0

RUN pip install --no-build-isolation mamba-ssm-macos==1.0.2

RUN pip install -r /app/requirements.txt

RUN python - <<'PY'
import gradio
import importlib

mamba_ssm = importlib.import_module("mamba_ssm")

print("gradio import OK:", gradio.__version__)
print("mamba_ssm import OK:", mamba_ssm)
PY

COPY . /app

EXPOSE 7860

CMD ["python", "app.py"]
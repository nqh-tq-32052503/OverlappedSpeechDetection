# ---- Base image: CUDA runtime for GPU inference ----
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps: Python, pip, ffmpeg (often useful with librosa), git (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ffmpeg git \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip setuptools wheel

# Python packages (install GPU build of onnxruntime)
# Equivalent to:
# !pip install -q museval resampy
# !pip uninstall -q -y onnxruntime
# !pip install -q -U onnxruntime-gpu

RUN pip install \
    torch==2.8.0 \
    torch-audiomentations==0.12.0 \
    torch_pitch_shift==1.2.5 \
    torchaudio==2.8.0 \
    torchmetrics==1.8.2 \
    pydantic==2.11.10 \
    pydantic-settings==2.11.0 \
    pydantic_core==2.33.2 \
    pyannote.audio==3.4.0 \
    librosa==0.11.0 

RUN python3 -m pip install \
        numpy \
        soundfile \
        resampy \
        museval \
        onnxruntime-gpu

RUN pip install python-multipart

RUN pip install \
    fastapi==0.115.4 uvicorn[standard]==0.32.0 

# Optional: working directory for your code
WORKDIR /app
# If you have project files, uncomment the next line to copy them in:
COPY . /app

EXPOSE 8029
# Quick sanity check on container start (prints available ORT providers)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8029"]

FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /content

RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir runpod requests torchaudio

RUN pip install --no-cache-dir cosyvoice2-eu

RUN pip install --no-cache-dir --force-reinstall torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

COPY worker_runpod.py /content/worker_runpod.py

CMD ["python3", "-u", "worker_runpod.py"]

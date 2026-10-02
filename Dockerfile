# AstroPhoto Studio web UI in a container: NVIDIA CUDA or CPU.
#
#   docker compose up -d --build                      # NVIDIA GPU (Linux / Windows + WSL2; default profile)
#   docker compose --profile cpu up -d --build        # CPU only (any host, including macOS)
#
# Containers on macOS cannot use the Apple GPU (Metal / MPS); run natively there for GPU speed.
# TORCH_INDEX picks the PyTorch build:
#   https://download.pytorch.org/whl/cu128   CUDA 12.8: RTX 20 ... 50 series (driver >= 570)
#   https://download.pytorch.org/whl/cu126   CUDA 12.6: older drivers (>= 560), GTX 10 series
#   https://download.pytorch.org/whl/cpu     CPU only (a much smaller image)
# The CUDA wheels bring their own CUDA runtime, so a slim base image is enough; the host needs
# only the NVIDIA driver and the NVIDIA Container Toolkit.
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # caches of astropy / astroquery / torch go to the output volume: writable by any UID, and kept
    HOME=/data/output/.home

# libgomp: OpenMP for NumPy/SciPy/OpenCV/torch on CPU
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# torch first, from the chosen index, so requirements.txt does not pull the default build
RUN pip install torch --index-url "${TORCH_INDEX}" \
 && pip install -r requirements.txt

COPY astrophoto ./astrophoto
COPY webui ./webui
COPY README.md ./

# /data/images: the session folders (mounted read-only), /data/output: caches and exports
RUN mkdir -p /data/images /data/output
VOLUME ["/data/output"]

# runs as root only to make the output volume writable, then as PUID:PGID (see the script)
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=8)"

CMD ["python", "-m", "webui.server", "--host", "0.0.0.0", "--port", "8000", \
     "--images", "/data/images", "--workdir", "/data/output"]

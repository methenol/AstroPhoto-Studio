# AstroPhoto Studio (sirilpipe backend): Siril + GraXpert + StarNet2 + RC-Astro CLI, everything inside the image.
#
#   docker compose up -d --build                      # NVIDIA GPU (GRAXPERT_EXTRA=cuda)
#   docker compose --profile cpu up -d --build        # CPU only   (GRAXPERT_EXTRA=cpuonly)
#
# Build args:
#   GRAXPERT_EXTRA          cuda | cpuonly   (cuda also brings the CUDA 12 user-space libs rc-astro needs)
#   STARNET_ACCEPT_LICENSE  "yes" = you accept the StarNet2 licence and the image downloads the official zip
#                           from download.starnetastro.com (not redistributable otherwise); empty = skipped,
#                           the app offers the download at run time into $HOME/tools/starnet2.
#   IMAGEMM                 off (default) | cuda | cpu - installs torch (+ torchvision, onnxruntime for Cosmic Clarity) for the optional
#                           ImageMM restoration (deconv_method = imagemm): cuda = the cu128 wheels (~7 GB with the
#                           NVIDIA libraries), cpu = the CPU wheels (~0.3 GB, slow).  off: the image has no torch and
#                           a request for ImageMM falls back to BlurXTerminator / Siril RL (recorded in restore_meta.json).
#   vendor/rc-astro-cli-*-linux-x64.sh   user-supplied RC-Astro installer (gitignored); optional.
# Nothing is taken from the host; no licence/config of any tool is ever baked in.

ARG GRAXPERT_EXTRA=cuda
ARG STARNET_ACCEPT_LICENSE=

# ---------------------------------------------------------------- Siril AppImage + script/SPCC repos
FROM python:3.12-slim AS siril-dl
ARG SIRIL_VER=1.4.4
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /dl
RUN curl -fsSL -o Siril.AppImage "https://free-astro.org/download/Siril-${SIRIL_VER}-x86_64.AppImage" \
 && chmod +x Siril.AppImage && ./Siril.AppImage --appimage-extract >/dev/null \
 && mv squashfs-root /opt/siril \
 && rm -rf /opt/siril/usr/share/icons /opt/siril/usr/share/locale
# Siril insists on the canonical remotes
RUN mkdir -p /opt/siril-data \
 && git clone --depth 1 https://gitlab.com/free-astro/siril-scripts.git      /opt/siril-data/siril-scripts \
 && git clone --depth 1 https://gitlab.com/free-astro/siril-spcc-database.git /opt/siril-data/siril-spcc-database

# ---------------------------------------------------------------- GraXpert venv (own numpy / onnxruntime)
FROM python:3.12-slim AS graxpert-build
ARG GRAXPERT_EXTRA
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/graxpert \
 && /opt/graxpert/bin/pip install "graxpert[${GRAXPERT_EXTRA}]==3.2.0a3"
# rc-astro needs cudart 12, cublas 12, cufft 11, cudnn 9: the graxpert[cuda] wheels carry them; add any missing
RUN if [ "${GRAXPERT_EXTRA}" = "cuda" ]; then \
      SP=$(/opt/graxpert/bin/python -c 'import site;print(site.getsitepackages()[0])'); \
      for pair in cuda_runtime:libcudart.so.12 cublas:libcublas.so.12 cufft:libcufft.so.11 cudnn:libcudnn.so.9; do \
        d=${pair%%:*}; f=${pair##*:}; \
        if ! ls "$SP/nvidia/$d/lib/$f" >/dev/null 2>&1; then echo "missing $f -> pip install"; \
          /opt/graxpert/bin/pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-cudnn-cu12; break; fi; \
      done; \
      for pair in cuda_runtime:libcudart.so.12 cublas:libcublas.so.12 cublas:libcublasLt.so.12 cufft:libcufft.so.11 cudnn:libcudnn.so.9; do \
        ls "$SP/nvidia/${pair%%:*}/lib/${pair##*:}" >/dev/null || exit 1; done; \
    fi
# GPU patch: GraXpert runs inference in a fork()ed child, which dies after CUDA init (tools.md A.6)
RUN printf '%s\n' \
 '"""GraXpert CLI with in-process inference so the CUDA provider works (fork() after CUDA init fails)."""' \
 'import graxpert.ai_model_handling as m' \
 'm.run_in_process = lambda fn: fn()' \
 'from graxpert.main import main' \
 'main()' > /opt/graxpert/gx_run.py
# strip bytecode caches / tests to keep the layer smaller
RUN find /opt/graxpert -name __pycache__ -type d -prune -exec rm -rf {} + 

# ---------------------------------------------------------------- StarNet2 (only with accepted licence)
FROM python:3.12-slim AS starnet-dl
ARG STARNET_ACCEPT_LICENSE
ARG STARNET_ZIP=starnet2_linux_2.6.2-0242_ORT_x64_cli.zip
RUN mkdir -p /opt/starnet2 /dl && if [ "${STARNET_ACCEPT_LICENSE}" = "yes" ]; then \
      python -c "import urllib.request as u; u.urlretrieve('https://download.starnetastro.com/${STARNET_ZIP}', '/dl/s.zip')" \
      && python -m zipfile -e /dl/s.zip /dl/x \
      && mv /dl/x/*/* /opt/starnet2/ \
      && chmod 755 /opt/starnet2/starnet2 \
      && rm -rf /dl; \
    else echo "StarNet2 skipped (STARNET_ACCEPT_LICENSE != yes)"; fi

# ---------------------------------------------------------------- RC-Astro CLI (user-supplied installer)
FROM python:3.12-slim AS rcastro
# requirements.txt always exists, so the glob copy succeeds even when vendor/ is absent
COPY requirements.txt vendor* /tmp/vendor/
RUN mkdir -p /opt/rc-bin /opt/rc-astro; \
    inst=$(ls /tmp/vendor/rc-astro-cli-*-linux-x64.sh 2>/dev/null | sort | tail -n 1); \
    if [ -n "$inst" ]; then \
      mkdir -p /tmp/rcx && sh "$inst" /tmp/rcx </dev/null \
      && ls /tmp/rcx/install.sh \
      && RCAC_INSTALL_PREFIX=/opt/rc-astro RCAC_BIN_DIR=/opt/rc-bin sh /tmp/rcx/install.sh; \
    else echo "RC-Astro installer not provided: vendor/rc-astro-cli-*-linux-x64.sh missing"; fi; \
    rm -rf /tmp/rcx /tmp/vendor

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim
ARG GRAXPERT_EXTRA

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/data/output/.home \
    LC_ALL=C.UTF-8 \
    APPDIR=/opt/siril \
    QT_QPA_PLATFORM=offscreen \
    SIRIL_BIN=/usr/local/bin/siril-cli \
    SIRIL_DATA=/opt/siril-data \
    GRAXPERT_BIN=/opt/graxpert/bin/python \
    GRAXPERT_RUNNER=/opt/graxpert/gx_run.py \
    STARNET_DIR=/opt/starnet2 \
    RCASTRO_BIN=/usr/local/bin/rc-astro \
    PATH=/opt/starnet2:$PATH

# libraries PyQt6 needs to be imported by the siril-scripts Python scripts run headless with `pyscript` (QT_QPA_PLATFORM=offscreen): Statistical_Stretch.py (Seti Astro) is
# the stretch method `statstretch` of sirilpipe/stretch.py and imports PyQt6.QtWidgets before it looks at its command line (libGL.so.1 was missing: ImportError, and a failing
# pyscript does not fail the Siril script).  The pyscript venv installs PyQt6 itself on first use (network once).
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libegl1 libxkbcommon0 libfontconfig1 libglib2.0-0 libdbus-1-3 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=siril-dl /opt/siril /opt/siril
COPY --from=siril-dl /opt/siril-data /opt/siril-data
RUN printf '#!/bin/sh\nexec /opt/siril/AppRun siril-cli "$@"\n' > /usr/local/bin/siril-cli \
 && chmod +x /usr/local/bin/siril-cli

COPY --from=graxpert-build /opt/graxpert /opt/graxpert
COPY --from=starnet-dl /opt/starnet2 /opt/starnet2
COPY --from=rcastro /opt/rc-astro /opt/rc-astro
COPY --from=rcastro /opt/rc-bin/ /usr/local/bin/

# libs the tool subprocesses need on LD_LIBRARY_PATH (nvidia wheels of the graxpert venv + the rc-astro dir).
# Only tool subprocesses get it (sirilpipe.config.tool_env()); the entrypoint exports nothing globally.
ENV TOOL_LD_LIBRARY_PATH=/opt/graxpert/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/graxpert/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/graxpert/lib/python3.12/site-packages/nvidia/cufft/lib:/opt/graxpert/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/rc-astro
RUN if [ "${GRAXPERT_EXTRA}" = "cuda" ]; then \
      for d in $(echo "$TOOL_LD_LIBRARY_PATH" | tr ':' ' '); do test -d "$d" || { echo "missing $d"; exit 1; }; done; \
    fi

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt \
 && python -c "import numpy, scipy, cv2, astropy, tifffile, PIL, fastapi, uvicorn; print('native libraries ok')"

# optional ImageMM: torch from the matching PyTorch index (its own layer, before the app code, so code changes
# do not reinstall it)
ARG IMAGEMM=off
RUN case "${IMAGEMM}" in \
      cuda) pip install torch --index-url https://download.pytorch.org/whl/cu128 ;; \
      cpu)  pip install torch --index-url https://download.pytorch.org/whl/cpu ;; \
      off|"") echo "IMAGEMM=off: torch not installed (deconv_method=imagemm falls back to BXT / RL)" ;; \
      *) echo "IMAGEMM must be cuda, cpu or off (got '${IMAGEMM}')"; exit 1 ;; \
    esac
# Cosmic Clarity (SetiAstro's models, run by siril-scripts' CosmicClarity_Native.py in Siril's pyscript environment, which is
# given this torch: sirilpipe/tools.CosmicClarity): torchvision for its satellite-trail detectors, pinned to the torch above
# through the same index, and onnxruntime for the models published as ONNX only (DarkStar).  onnxruntime-gpu 1.23 is the last
# CUDA 12 build: it uses the CUDA 12.8 / cuDNN 9 wheels torch brought (1.24+ want CUDA 13 and fall back to the CPU).
RUN case "${IMAGEMM}" in \
      cuda) pip install "torch==$(python -c 'import torch; print(torch.__version__)')" torchvision --index-url https://download.pytorch.org/whl/cu128 \
            && pip install onnxruntime-gpu==1.23.2 ;; \
      cpu)  pip install "torch==$(python -c 'import torch; print(torch.__version__)')" torchvision --index-url https://download.pytorch.org/whl/cpu \
            && pip install onnxruntime==1.23.2 ;; \
      *)    echo "IMAGEMM=off: no torch, Cosmic Clarity unavailable" ;; \
    esac

COPY sirilpipe ./sirilpipe
COPY astrophoto ./astrophoto
COPY webui ./webui
COPY tools ./tools
COPY README.md ./

# /data/images: the session folders (mounted read-only), /data/output: caches, licences, models, exports
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

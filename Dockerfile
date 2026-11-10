# One image for both services in compose.yaml: JupyterLab for the notebooks, and the pipeline that
# rebuilds every number, model, chart and notebook. Same Python, same libraries, same results.
FROM python:3.12.15-slim

# `make` runs the pipeline steps. Nothing else is needed from the operating system.
RUN apt-get update \
 && apt-get install -y --no-install-recommends make \
 && rm -rf /var/lib/apt/lists/*

# PY and DATA_ZIP tell the Makefile which Python to use and where the supplied data.zip is mounted.
# HOME and MPLCONFIGDIR point at /tmp so the services can run as your user, not as root.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PY=python \
    DATA_ZIP=/input/data.zip \
    MPLBACKEND=Agg \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Libraries first, at the exact versions the project was tested with. Docker caches this layer, so
# editing the code doesn't reinstall them.
COPY requirements.lock .
RUN pip install -r requirements.lock

# Then the project itself, installed in editable mode: when compose mounts this folder over /app,
# the services run whatever code is on your machine right now. JupyterLab's news pop-up is switched
# off, so opening the notebooks never asks to fetch anything from the internet.
COPY . .
RUN pip install --no-deps -e . \
 && jupyter labextension disable "@jupyterlab/apputils-extension:announcements"

EXPOSE 8888
CMD ["make", "all"]

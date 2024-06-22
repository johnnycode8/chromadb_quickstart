# Image reference: https://github.com/chroma-core/chroma/releases
FROM ghcr.io/chroma-core/chroma:0.5.2

COPY requirements.txt .
RUN pip install --upgrade pip

# Install dependencies from requirements file. Turn off pip's cache as docker containers will not make use of it.
RUN pip install --no-cache-dir -r requirements.txt
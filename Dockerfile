FROM python:3.11-slim-bookworm

WORKDIR /usr/src/app
COPY . .
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    build-essential
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e .
CMD ["roofs", "db", "status"]
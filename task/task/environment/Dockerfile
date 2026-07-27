FROM public.ecr.aws/docker/library/python@sha256:REPLACE_WITH_PINNED_DIGEST_FROM_check-base-image.sh

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.1 \
    pandas==2.2.2 \
    pytest==8.2.2 \
    pytest-json-ctrf==0.3.3

COPY environment/data/sensor_signal.csv /app/data/sensor_signal.csv

RUN mkdir -p /app/output

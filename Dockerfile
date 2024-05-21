# Stage 1: Build the Airflow image
FROM apache/airflow:2.8.1 AS airflow

# Install additional Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install dvc

# Stage 2: Build the Flask image
FROM python:3.9-slim-buster AS flask

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libc-dev \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN apt-get update && pip install -r requirements.txt
CMD ["python", "main.py"]
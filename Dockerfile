# Use the official Airflow image as the base
FROM apache/airflow:2.8.1

# Install additional Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install dvc

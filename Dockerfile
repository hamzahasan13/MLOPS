## Python base-image from docker hub
FROM python:3.9 

## User
USER root

## Creating app directory
RUN mkdir /app

## Copying all the contents of the current directory to app directory
COPY . /app/

## Making app the current directory
WORKDIR /app/

## Installing dependencies
RUN pip install -r requirements.txt

## Install Airflow and its dependencies
RUN pip install apache-airflow

## Setting Environment
ENV AIRFLOW_HOME = "/app/airflow"
ENV AIRFLOW_CORE_DAGBAG_IMPORT_TIMEOUT = 1000
ENV AIRFLOW_CORE_ENABLE_XCOM_PICKLING = True
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor

## Running Airflow and storing meta data in db
RUN airflow db init
#RUN airflow users create -e hamzahasan0713@gmail.com -f Hamza -l Farooqi -p admin -r Admin -u admin

## Giving access to start.sh
RUN chmod + start.sh

## Updating system
RUN apt update -y

## Entry point
ENTRYPOINT ["/bin/sh"]

CMD ["start.sh"]

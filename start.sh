#!/bin/sh

## Activates airflow scheduler and webserver
nohup airflow scheduler & 
airflow webserver
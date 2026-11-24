# Airflow image (webserver / scheduler / worker) with the ML dependencies baked in,
# so DAG tasks can import src/ without pip-installing at container start.
FROM apache/airflow:2.10.4-python3.10

COPY airflow/requirements.txt /tmp/requirements-airflow.txt
# Install against Airflow's constraint file so provider versions stay compatible.
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.4/constraints-3.10.txt"

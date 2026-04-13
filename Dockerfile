FROM apache/spark-py:latest
USER root
RUN pip install --no-cache-dir opencv-python-headless numpy pandas pyarrow
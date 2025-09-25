FROM public.ecr.aws/docker/library/python:3.9-slim
# FROM python:3.9-slim

WORKDIR /opt/ml/processing

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt code/
RUN pip install --no-cache-dir -r code/requirements.txt

# Copy source code
COPY src/ code/src/

# Set Python path
ENV PYTHONPATH="/opt/ml/processing/code:${PYTHONPATH}"

# Set entrypoint
ENTRYPOINT ["python", "/opt/ml/processing/code/src/main.py"]

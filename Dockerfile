# The Official Upstream PyTorch Image (v2.5.1 with CUDA 12.4)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Install basic Linux visual libraries for image processing
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install your requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python logic
COPY handler.py .

# Hard-link the endpoint to your pre-downloaded Network Volume
ENV MODEL_PATH=/runpod-volume/models/FLUX.1-dev \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1

# Start the serverless worker
CMD ["python", "-u", "handler.py"]
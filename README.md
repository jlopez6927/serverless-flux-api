# Serverless FLUX.1-dev API Deployment
Deploy [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) as a RunPod serverless endpoint using a highly optimized, decoupled Network Volume architecture.

## Project Overview
This repository contains my deployment of a serverless API for **FLUX.1-dev**, a large 24GB text-to-image AI model. 

My main goal for this project was to get a 24GB AI model running efficiently in the cloud without dealing with massive 5+ minute load times every time the server woke up.

Instead of baking the 24GB model directly into a Docker image (which causes terrible container pull times), I set up a decoupled architecture. I downloaded the model once to a persistent RunPod Network Volume, and then connected a lightweight custom Docker container to that volume. Now, the model loads directly into the GPU's memory in about 85 seconds from a cold start, and scales down to $0.00 when not in use.

## Repository Structure
```text
.
├── handler.py          ← RunPod serverless handler (inference logic)
├── Dockerfile          ← Lightweight build (pulls official PyTorch 2.5.1)
├── requirements.txt    ← Python dependencies (pinned to avoid math errors)
├── test_endpoint.py    ← Local test script (prompts user and saves timestamped image)
└── .gitignore          ← Protects API keys and ignores generated test images
```

## Prerequisites
Requirement	              Notes
Docker Desktop	          Required to build and push the image
Docker Hub account	      Free tier is fine
HuggingFace account	      Must accept the FLUX.1-dev licence
HuggingFace access token	read scope is sufficient
RunPod account	          For creating the Network Volume and Serverless Endpoint

## Deployment Steps

### Step 1 – Accept the FLUX.1-dev licence

Visit [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) on Hugging Face and click "Agree and access repository". Then create a read token in your account settings.

### Step 2 – Create the Network Volume (The Storage)

1. Go to the Storage tab in RunPod and create a 50GB Network Volume in your preferred region.
2. Deploy a temporary Pod in that same region and attach the new Network Volume.
3. Open the Web Terminal in the Pod and run this script to download the model efficiently:

pip install huggingface_hub

export HF_TOKEN="your_huggingface_token"
```text
cat << 'EOF' > download_model.py
import os
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN")
snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    local_dir="/workspace/models/FLUX.1-dev",
    token=token,
    ignore_patterns=["flux1-dev.safetensors", "ae.safetensors"]
)
EOF
```
python download_model.py

4. Once the download finishes, terminate this temporary Pod.

### Step 3 – Build & Push the Docker Image

Open your terminal in this repository's folder and build the lightweight Docker image:

docker login
docker build -t yourdockerhubname/runpod-flux-network:v1 .
docker push yourdockerhubname/runpod-flux-network:v1

### Step 4 – Create a RunPod Serverless Endpoint

1. Go to Serverless → + New Endpoint in the RunPod console.
2. Configure a customized endpoint with your Docker Hub image (yourdockerhubname/runpod-flux-network:v1) and attach your 50GB Network Volume.
3. Check A100 PCIe, A100 SXM, and H100 under compatible GPUs.
4. Set Min Workers to 0 and click Save Endpoint.

### Step 5 – Test the Endpoint

Add your RunPod API key and Endpoint ID to test_endpoint.py, then run it locally:

python test_endpoint.py

## Architecture & Strategy

Deploying a model this heavy required some trial and error to get the storage and hardware working together. Here is the approach I took:

### 1. Fixing the "Cold Start" Download Issue
* **The Problem:** Standard setups try to download the whole model from Hugging Face every time the API receives a request. 
* **The Solution:** I created a 50GB Network Volume in RunPod. I ran a temporary pod just to download the model into this volume. Then, I set my main serverless endpoint to read directly from this drive using the `MODEL_PATH` variable. This skips the download step completely.

### 2. Optimizing the Storage Space
* **The Problem:** My first attempt at downloading the model completely filled up my 50GB drive and crashed the pod.
* **The Solution:** I realized Hugging Face stores two versions of the model in the same repository (the folder pieces I needed, and a massive standalone `.safetensors` file meant for other software). I used an `ignore_patterns` script during the download to filter out the standalone files I didn't need, which kept my storage use at a clean ~24GB.

### 3. Choosing the Right Base Image
* **The Problem:** When trying to run the model, my code kept crashing with an error: `AttributeError: module 'torch.nn' has no attribute 'RMSNorm'`.
* **The Solution:** After some research, I found out the basic PyTorch version I was using was missing some functions needed for this specific model. I updated my `Dockerfile` to use a newer, official image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`. This fixed the compatibility issues immediately.

## The Codebase

* **`Dockerfile`**: Pulls the official PyTorch 2.5.1 image, installs the dependencies from `requirements.txt`, and sets environment variables so Hugging Face knows to look at my local Network Volume instead of downloading from the internet.
* **`handler.py`**: The main Python script. It loads the model into the GPU, processes the prompt, and returns the generated image as a Base64 encoded string.
* **`requirements.txt`**: The specific library versions needed to run the `diffusers` pipeline.
* **`test_endpoint.py`**: A local script I use to send the API request to RunPod, allowing the user to type a custom prompt, and decodes the returning Base64 string back into a uniquely named `.png` file.

## Challenges & Troubleshooting

Getting this to work smoothly taught me a lot about cloud GPU environments. Here were the biggest hurdles:

**1. "Cloud Whales" and GPU Availability**
While setting up my Network Volume, I found that high-end GPUs like the H100 are in massive demand and often totally unavailable in certain regions. Because my storage and my code were separated, I was able to easily migrate my setup to a different EU region that had available A100 GPUs without having to rewrite my Docker image.

**2. The RTX 5090 / Blackwell Compatibility Error**
To avoid long queue times, I initially configured my endpoint to accept any GPU with enough memory, including the brand new RTX 5090. However, the script crashed with this error:
`NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.`
* *How I fixed it:* I learned that the RTX 5090 architecture is so new that the stable versions of PyTorch don't fully support it yet. I went back into my endpoint settings and explicitly unselected the 5090, restricting my deployment to older, stable enterprise cards like the A100 and H100.

**3. VRAM Overflow (Out of Memory)**
I also ran into `CUDA out of memory` errors when the endpoint tried to run on a 24GB card. I learned that even though the model is 24GB, it needs extra memory overhead to actually do the math to generate the image. Pinning the endpoint to 80GB cards (like the A100) completely solved this.

## Future Improvements
* Add better `print()` or logging statements inside `handler.py` to track exactly how long the queue delays are vs. the actual image generation time.

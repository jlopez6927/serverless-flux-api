import runpod
import torch
import base64
import os
import time
import logging
from io import BytesIO
from diffusers import FluxPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/models/FLUX.1-dev")
pipe = None

def load_model():
    global pipe
    if pipe is not None:
        return

    logger.info(f"Loading FLUX.1-dev from Network Volume: {MODEL_PATH}")
    t0 = time.time()

    pipe = FluxPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")

    logger.info(f"Pipeline loaded in {time.time() - t0:.1f} seconds")

load_model()

def handler(job: dict) -> dict:
    job_input = job.get("input", {})
    prompt = job_input.get("prompt", "Monkey D. Luffy standing on the deck of the Thousand Sunny, cinematic lighting, 8k")
    
    if pipe is None:
        return {"error": "Pipeline failed to load from Network Volume."}

    t0 = time.time()
    try:
        result = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=28,
            guidance_scale=3.5,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0)
        )
        elapsed = time.time() - t0

        image = result.images[0]
        buf = BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {"image": b64, "inference_time": round(elapsed, 2)}
        
    except Exception as exc:
        return {"error": str(exc)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
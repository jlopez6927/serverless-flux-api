import requests
import base64
import time

API_KEY = "YOUR_RUNPOD_API_KEY"
ENDPOINT_ID = "YOUR_ENDPOINT_ID"

url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# This asks you to type your prompt in the terminal
user_prompt = input("🏴‍☠️ Enter your prompt for FLUX: ")

payload = {
    "input": {
        "prompt": user_prompt
    }
}

print("🚀 Waking up the cloud and generating your image...")

response = requests.post(url, json=payload, headers=headers)
result = response.json()

if 'output' in result and 'image' in result['output']:
    image_data = base64.b64decode(result['output']['image'])
    
    # Creates a unique filename using the current time
    timestamp = int(time.time())
    file_name = f"generated_image_{timestamp}.png"
    
    with open(file_name, "wb") as f:
        f.write(image_data)
        
    print(f"✅ Success! Image saved to your folder as '{file_name}'")
    print(f"⏱️ GPU Inference Time: {result['output'].get('inference_time', 'N/A')} seconds")
else:
    print("❌ Something went wrong:")
    print(result)
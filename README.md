# IDM-VTON — RunPod Serverless Endpoint

Deploy the [IDM-VTON](https://github.com/yisol/IDM-VTON) virtual try-on model as a REST API on [RunPod Serverless](https://www.runpod.io/serverless-gpu).

Send a person photo + a garment photo → receive a generated try-on result image.

---

## Table of Contents

- [How It Works](#how-it-works)
- [API: Input / Output](#api-input--output)
- [Environment Variables](#environment-variables)
- [GPU Requirements](#gpu-requirements)
- [Cold Start Behaviour](#cold-start-behaviour)
- [Deploying on RunPod](#deploying-on-runpod)
- [Testing the Endpoint](#testing-the-endpoint)
- [Optimisation Options](#optimisation-options)
- [File Structure](#file-structure)

---

## How It Works

1. **Docker container starts** → `start.sh` runs first.
2. `start.sh` downloads model weights (IDM-VTON from HuggingFace, DensePose checkpoint, OpenPose, human parsing ONNX models) into `/workspace/`.
3. `handler.py` loads all models into GPU memory (once per cold start).
4. RunPod calls `handler(event)` for each API request.
5. The handler decodes the base64 images, runs the try-on pipeline, and returns the result image as base64.

---

## API: Input / Output

### Request body (`input` field)

```json
{
  "input": {
    "person_image":        "<base64-encoded image string>",
    "garment_image":       "<base64-encoded image string>",
    "garment_description": "blue cotton button-down shirt",
    "category":            "upper_body",
    "denoise_steps":       30,
    "seed":                42
  }
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `person_image` | string (base64) | ✅ Yes | — | Photo of the person. Accepts JPEG, PNG, WebP. Data-URL prefix (`data:image/...;base64,`) is stripped automatically. |
| `garment_image` | string (base64) | ✅ Yes | — | Photo of the garment on a plain background. |
| `garment_description` | string | No | `"garment"` | Short text description of the garment, e.g. `"red floral summer dress"`. More specific = slightly better results. |
| `category` | string | No | `"upper_body"` | One of `"upper_body"`, `"lower_body"`, or `"dresses"`. Determines which body region is masked for inpainting. |
| `denoise_steps` | integer | No | `30` | Number of diffusion steps. Range: 20–40. More steps = better quality but slower (clamped to 40 max). |
| `seed` | integer | No | `42` | Random seed for reproducibility. |

### Response body (success)

```json
{
  "output": {
    "result_image": "<base64-encoded PNG string>",
    "status": "success"
  }
}
```

### Response body (error)

```json
{
  "output": {
    "status": "error",
    "error": "Human-readable error message or Python traceback"
  }
}
```

---

## Environment Variables

Set these in the RunPod Serverless endpoint settings under **Environment Variables**:

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | No | Your HuggingFace token. Only needed if you want to download private / gated models. IDM-VTON itself is public, so this is optional. |
| `HF_HOME` | No | Cache directory for HuggingFace downloads. Default: `/workspace/hf_cache`. If you mount a Network Volume, point this there to persist the cache. |

> **Tip:** If you mount a RunPod Network Volume to `/workspace`, set `HF_HOME=/workspace/hf_cache` and the model weights will persist across worker restarts, dramatically improving cold-start times after the first run.

---

## GPU Requirements

| GPU | VRAM | Status |
|---|---|---|
| RTX A5000 | 24 GB | ✅ Recommended — comfortable headroom |
| RTX A4000 | 16 GB | ✅ Should work with float16 |
| L40S | 48 GB | ✅ Excellent — plenty of headroom |
| RTX 3090 | 24 GB | ✅ Should work |
| RTX 4090 | 24 GB | ✅ Fastest consumer GPU option |
| T4 | 16 GB | ⚠️ Tested working on Colab; tight VRAM — reduce `denoise_steps` to 20 if OOM |
| A10G | 24 GB | ✅ Should work |

**Minimum recommended: 16 GB VRAM.** The pipeline uses float16 precision throughout to minimise VRAM usage. The model itself (UNet × 2, VAE, text encoders, image encoder) requires approximately 12–14 GB of VRAM, leaving headroom for intermediate activations.

**Container disk size: minimum 30 GB** — the model weights (~7 GB), DensePose (~1 GB), and the Docker image layers together require at least 25 GB.

---

## Cold Start Behaviour

**Cold start = the first request after the worker was idle (or never ran before).**

On a cold start, the container:
1. Pulls the Docker image (~8–10 GB, cached by RunPod after first pull)
2. Runs `start.sh` which downloads model weights:
   - IDM-VTON weights from HuggingFace: ~7 GB — **~3–5 min on first run**
   - DensePose checkpoint: ~900 MB
   - Human parsing ONNX models: ~200 MB
   - OpenPose body model: ~200 MB
3. Loads all models into GPU memory: ~60–90 seconds

**Total first cold start: ~5–8 minutes.**

**Subsequent cold starts (with Network Volume caching):** ~60–90 seconds (weights already cached, just loading into GPU).

**Warm requests (while worker is active):** ~10–20 seconds per image (30 denoising steps on an A5000).

### How to minimise cold starts

1. **Mount a Network Volume** (50 GB minimum) to `/workspace` — weights persist between restarts.
2. Set **Min Workers = 1** in RunPod — keeps one worker always warm (costs ~$0.50–1.00/hr depending on GPU).
3. Use **FlashAttention** (see Optimisation section) to reduce per-request time.

---

## Deploying on RunPod

### Step 1: Push this repo to GitHub

```bash
cd idm-vton-runpod
git init
git add .
git commit -m "Initial IDM-VTON RunPod Serverless setup"
git remote add origin https://github.com/YOUR_USERNAME/idm-vton-runpod.git
git push -u origin main
```

### Step 2: Create a RunPod account and connect GitHub

1. Go to [runpod.io](https://runpod.io) and sign up / log in.
2. Go to **Serverless** → **New Endpoint**.
3. Click **"Deploy from a GitHub repository"**.
4. Authenticate GitHub and select your `idm-vton-runpod` repository.

### Step 3: Configure the endpoint

| Setting | Recommended value |
|---|---|
| **GPU Type** | RTX A5000 (24 GB) — best price/performance for this model |
| **Container Disk** | 30 GB minimum |
| **Network Volume** | 50 GB — mount to `/workspace` to persist model cache |
| **Min Workers** | 0 (scale to zero when idle) or 1 (always warm) |
| **Max Workers** | 3–5 (adjust based on expected load) |
| **Idle Timeout** | 30 seconds (time before worker scales down) |
| **Execution Timeout** | 300 seconds (5 minutes — enough for cold start + inference) |

### Step 4: Set environment variables (optional)

In the endpoint settings, add:
- `HF_TOKEN` = your HuggingFace token (only if needed for private models)

### Step 5: Deploy and test

Click **Deploy**. RunPod will build the image, push it to its registry, and provision a worker. Use the **Test** tab in the RunPod console to send a sample request.

---

## Testing the Endpoint

### Using RunPod's web console

In the endpoint's **Test** tab, paste:

```json
{
  "input": {
    "person_image": "<paste base64 image here>",
    "garment_image": "<paste base64 image here>",
    "garment_description": "white linen shirt",
    "category": "upper_body"
  }
}
```

### Using Python

```python
import runpod
import base64
from pathlib import Path

runpod.api_key = "YOUR_RUNPOD_API_KEY"

def image_to_base64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")

endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

result = endpoint.run_sync({
    "input": {
        "person_image":        image_to_base64("person.jpg"),
        "garment_image":       image_to_base64("shirt.jpg"),
        "garment_description": "white linen shirt",
        "category":            "upper_body",
        "denoise_steps":       30,
        "seed":                42,
    }
}, timeout=300)

# Decode and save the result image
output_b64 = result["output"]["result_image"]
image_bytes = base64.b64decode(output_b64)
with open("result.png", "wb") as f:
    f.write(image_bytes)

print("Saved result.png")
```

### Using curl

```bash
curl -X POST \
  "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "person_image": "'$(base64 -i person.jpg | tr -d '\n')'",
      "garment_image": "'$(base64 -i shirt.jpg | tr -d '\n')'",
      "garment_description": "white linen shirt",
      "category": "upper_body"
    }
  }'
```

---

## Optimisation Options

### Option 1: xFormers memory-efficient attention (easy win)

Reduces VRAM usage by ~20% and speeds up inference ~10–15%.

Add to `requirements.txt`:
```
xformers
```

Add to `handler.py` after `pipe.to(DEVICE)`:
```python
pipe.enable_xformers_memory_efficient_attention()
```

### Option 2: CPU offloading (for GPUs with tight VRAM)

Add to `handler.py` instead of `pipe.to(DEVICE)`:
```python
pipe.enable_model_cpu_offload()
```
This keeps models on CPU and moves them to GPU only when needed. **Slower** (adds ~30% inference time), but works on 16 GB GPUs with confidence.

### Option 3: Reduce output resolution

Change the resize targets in `handler.py` from `(768, 1024)` to `(512, 768)` for a smaller output. **Faster** (~2×) but lower quality.

### Option 4: Quantisation (INT8 — advanced)

The pipeline can be quantised to INT8 using `bitsandbytes`, but this requires careful testing with IDM-VTON's custom UNet code. Not recommended unless you have specific VRAM constraints.

---

## File Structure

```
idm-vton-runpod/
├── handler.py        # RunPod Serverless handler (main entry point)
├── Dockerfile        # Container build instructions
├── requirements.txt  # Python dependencies
├── start.sh          # Startup script: downloads weights, launches handler
└── README.md         # This file
```

The IDM-VTON model source code is cloned from HuggingFace directly into the container during the Docker build (see Dockerfile). You do not need to copy those files into this repo.

---

## Credits

- **IDM-VTON model**: [Yisol Choi et al.](https://github.com/yisol/IDM-VTON) — [HuggingFace Space](https://huggingface.co/spaces/yisol/IDM-VTON)
- **DensePose**: [Facebook AI Research](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose)
- **RunPod Serverless**: [runpod.io](https://www.runpod.io)

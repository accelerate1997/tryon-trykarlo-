"""
handler.py — RunPod Serverless entry point for IDM-VTON Virtual Try-On

How RunPod Serverless works:
  - When the container first starts (cold start), all the code at module level
    runs once. We use this to load the heavy ML models into GPU memory.
  - After that, for every API request RunPod calls `handler(event)`.
  - `event["input"]` is the JSON body your caller sends.
  - Whatever this function returns becomes the JSON response body.
"""

import os
import sys
import io
import base64
import traceback

import runpod  # RunPod Serverless SDK — must be installed via pip

# ---------------------------------------------------------------------------
# 0.  Environment tweaks — do these BEFORE importing torch/diffusers
# ---------------------------------------------------------------------------

# Prevent CUDA OOM from memory fragmentation (common with large diffusion
# pipelines running alongside Detectron2/DensePose).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

# DensePose stores its compiled extensions relative to the detectron2 repo.
# We cloned the repo to /workspace/detectron2 in the Dockerfile.
DETECTRON2_REPO = "/workspace/detectron2"
DENSEPOSE_PROJ  = os.path.join(DETECTRON2_REPO, "projects", "DensePose")
for p in [DETECTRON2_REPO, DENSEPOSE_PROJ]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# 1.  Standard imports — order matters: torch before anything CUDA-dependent
# ---------------------------------------------------------------------------

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer,
)
from diffusers import DDPMScheduler, AutoencoderKL

# IDM-VTON source files — these are copied into /workspace/IDM-VTON/ by the
# Dockerfile (they come from the HuggingFace Space's git repo).
IDM_VTON_DIR = "/workspace/IDM-VTON"
if IDM_VTON_DIR not in sys.path:
    sys.path.insert(0, IDM_VTON_DIR)

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from utils_mask import get_mask_location
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

# Detectron2 helpers used for DensePose body estimation
from detectron2.data.detection_utils import (
    convert_PIL_to_numpy,
    _apply_exif_orientation,
)
import apply_net  # lives in IDM-VTON root; calls DensePose under the hood

# ---------------------------------------------------------------------------
# 2.  Cold-start model loading
#     Everything here runs ONCE when the container starts, not per request.
#     This is the key trick for RunPod Serverless performance: you pay for
#     cold-start time only on the first request (or after idle scale-down).
# ---------------------------------------------------------------------------

print("[IDM-VTON] Loading models — this happens once per container start …")

BASE_MODEL_ID = "yisol/IDM-VTON"   # HuggingFace model repo
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.float16       # float16 halves VRAM usage vs float32

# --- UNet (the main denoiser that's been fine-tuned for virtual try-on) ---
unet = UNet2DConditionModel.from_pretrained(
    BASE_MODEL_ID,
    subfolder="unet",
    torch_dtype=DTYPE,
)
unet.requires_grad_(False)

# --- Tokenisers (text -> token IDs, used for garment description prompts) ---
tokenizer_one = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID, subfolder="tokenizer", use_fast=False
)
tokenizer_two = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID, subfolder="tokenizer_2", use_fast=False
)

# --- Noise scheduler (controls the denoising diffusion steps) ---
noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_ID, subfolder="scheduler")

# --- Text encoders (converts the garment description into embeddings) ---
text_encoder_one = CLIPTextModel.from_pretrained(
    BASE_MODEL_ID, subfolder="text_encoder", torch_dtype=DTYPE
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    BASE_MODEL_ID, subfolder="text_encoder_2", torch_dtype=DTYPE
)

# --- Image encoder (encodes the garment image for visual conditioning) ---
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    BASE_MODEL_ID, subfolder="image_encoder", torch_dtype=DTYPE
)

# --- VAE (encodes/decodes between pixel space and latent space) ---
vae = AutoencoderKL.from_pretrained(BASE_MODEL_ID, subfolder="vae", torch_dtype=DTYPE)

# --- Garment encoder (a second UNet that encodes the garment appearance) ---
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    BASE_MODEL_ID, subfolder="unet_encoder", torch_dtype=DTYPE
)

# Disable gradients — we only do inference, never training
for model in [UNet_Encoder, image_encoder, vae, unet, text_encoder_one, text_encoder_two]:
    model.requires_grad_(False)

# --- Assemble the full try-on pipeline ---
pipe = TryonPipeline.from_pretrained(
    BASE_MODEL_ID,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=DTYPE,
)
pipe.unet_encoder = UNet_Encoder

# Move everything to GPU now (keeps cold-start memory usage predictable)
pipe.to(DEVICE)
pipe.unet_encoder.to(DEVICE)

# --- Human parsing model (segments the person into body-part regions) ---
# GPU index 0 — the only GPU on a RunPod serverless worker
parsing_model = Parsing(0)

# --- OpenPose model (detects body keypoints for mask generation) ---
openpose_model = OpenPose(0)
openpose_model.preprocessor.body_estimation.model.to(DEVICE)

# --- Image normalisation transform (same as the original app.py) ---
tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

print("[IDM-VTON] All models loaded and ready")

# ---------------------------------------------------------------------------
# 3.  Helper utilities
# ---------------------------------------------------------------------------

def decode_base64_image(b64_string: str) -> Image.Image:
    """
    Convert a base64-encoded image string to a PIL Image.
    Accepts strings with or without the data-URL prefix
    (e.g. 'data:image/png;base64,...').
    """
    if "," in b64_string:
        # Strip the 'data:image/xxx;base64,' prefix if present
        b64_string = b64_string.split(",", 1)[1]
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def encode_pil_to_base64(pil_image: Image.Image, fmt: str = "PNG") -> str:
    """
    Convert a PIL Image to a base64-encoded string (PNG by default).
    The caller can change fmt to 'JPEG' for smaller payloads.
    """
    buffer = io.BytesIO()
    pil_image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pil_to_binary_mask(pil_image, threshold=0):
    """
    Convert a PIL Image to a binary mask (0 or 255).
    Used when the caller supplies a custom mask instead of using auto-masking.
    """
    np_image = np.array(pil_image.convert("L"))
    binary  = (np_image > threshold).astype(np.uint8) * 255
    return Image.fromarray(binary)


# ---------------------------------------------------------------------------
# 4.  Core inference function
# ---------------------------------------------------------------------------

def run_tryon(
    person_image: Image.Image,
    garment_image: Image.Image,
    garment_description: str,
    category: str = "upper_body",
    denoise_steps: int = 30,
    seed: int = 42,
) -> Image.Image:
    """
    Run the IDM-VTON try-on pipeline.

    Args:
        person_image:        PIL image of the person (any size - we'll resize).
        garment_image:       PIL image of the garment (any size - we'll resize).
        garment_description: Short text description, e.g. 'blue cotton shirt'.
        category:            One of 'upper_body', 'lower_body', 'dresses'.
        denoise_steps:       Number of diffusion denoising steps (20-40).
                             More steps = better quality, but slower.
        seed:                Random seed for reproducibility.

    Returns:
        PIL Image with the virtual try-on result.
    """

    # Validate category
    valid_categories = {"upper_body", "lower_body", "dresses"}
    if category not in valid_categories:
        raise ValueError(f"category must be one of {valid_categories}, got '{category}'")

    # --- Resize inputs to the resolution IDM-VTON was trained on ---
    garment_img  = garment_image.convert("RGB").resize((768, 1024))
    human_img    = person_image.convert("RGB").resize((768, 1024))

    # --- Auto-generate the inpainting mask using OpenPose + human parsing ---
    # Runs at half resolution (384x512) for speed, then upscales the mask.
    keypoints    = openpose_model(human_img.resize((384, 512)))
    model_parse, _ = parsing_model(human_img.resize((384, 512)))
    mask, mask_gray = get_mask_location("hd", category, model_parse, keypoints)
    mask = mask.resize((768, 1024))

    # Greyed-out masked region (for debugging — not returned to caller)
    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    # --- DensePose: generate dense body-surface map (used as pose conditioning) ---
    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    # apply_net.py wraps the DensePose Detectron2 predictor.
    # 'dp_segm' = dense pose segmentation mode; '-v' = visualise.
    args = apply_net.create_argument_parser().parse_args((
        "show",
        "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",   # DensePose config
        "./ckpt/densepose/model_final_162be9.pkl",       # DensePose checkpoint
        "dp_segm", "-v",
        "--opts", "MODEL.DEVICE", DEVICE,
    ))
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]                       # BGR to RGB
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    # --- Text conditioning ---
    prompt_wearing   = f"model is wearing {garment_description}"
    prompt_garment   = f"a photo of {garment_description}"
    negative_prompt  = "monochrome, lowres, bad anatomy, worst quality, low quality"

    # --- Run the diffusion pipeline (all inside no_grad for efficiency) ---
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            # Encode "model is wearing ..." prompt
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt_wearing,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            # Encode "a photo of ..." garment reference prompt
            (prompt_embeds_c, _, _, _) = pipe.encode_prompt(
                [prompt_garment],
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=[negative_prompt],
            )

            # Prepare tensor inputs
            pose_tensor  = tensor_transform(pose_img).unsqueeze(0).to(DEVICE, DTYPE)
            garm_tensor  = tensor_transform(garment_img).unsqueeze(0).to(DEVICE, DTYPE)
            generator    = torch.Generator(DEVICE).manual_seed(seed)

            # The main diffusion call
            images = pipe(
                prompt_embeds=prompt_embeds.to(DEVICE, DTYPE),
                negative_prompt_embeds=negative_prompt_embeds.to(DEVICE, DTYPE),
                pooled_prompt_embeds=pooled_prompt_embeds.to(DEVICE, DTYPE),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(DEVICE, DTYPE),
                num_inference_steps=denoise_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_tensor,
                text_embeds_cloth=prompt_embeds_c.to(DEVICE, DTYPE),
                cloth=garm_tensor,
                mask_image=mask,
                image=human_img,
                height=1024,
                width=768,
                ip_adapter_image=garment_img.resize((768, 1024)),
                guidance_scale=2.0,
            )[0]

    return images[0]  # Return the first (and only) generated image

# ---------------------------------------------------------------------------
# 5.  RunPod handler function
#     This is the function RunPod calls for every incoming API request.
# ---------------------------------------------------------------------------

def handler(event: dict) -> dict:
    """
    RunPod Serverless handler.

    Expected event["input"]:
    {
        "person_image":        "<base64 string>",        # required
        "garment_image":       "<base64 string>",        # required
        "garment_description": "blue cotton shirt",      # optional
        "category":            "upper_body",             # optional
        "denoise_steps":       30,                       # optional (20-40)
        "seed":                42                        # optional
    }

    Returns:
    {
        "result_image": "<base64 PNG string>",
        "status": "success"
    }
    or on error:
    {
        "error": "<message>",
        "status": "error"
    }
    """
    try:
        job_input = event.get("input", {})

        # ---- Validate required inputs ----
        person_b64  = job_input.get("person_image")
        garment_b64 = job_input.get("garment_image")

        if not person_b64:
            return {"status": "error", "error": "Missing required field: 'person_image'"}
        if not garment_b64:
            return {"status": "error", "error": "Missing required field: 'garment_image'"}

        # ---- Decode images ----
        try:
            person_image = decode_base64_image(person_b64)
        except Exception as e:
            return {"status": "error", "error": f"Failed to decode 'person_image': {e}"}

        try:
            garment_image = decode_base64_image(garment_b64)
        except Exception as e:
            return {"status": "error", "error": f"Failed to decode 'garment_image': {e}"}

        # ---- Optional parameters with defaults ----
        garment_description = job_input.get("garment_description", "garment")
        category            = job_input.get("category", "upper_body")
        denoise_steps       = int(job_input.get("denoise_steps", 30))
        seed                = int(job_input.get("seed", 42))

        # Clamp steps to a sensible range
        denoise_steps = max(20, min(40, denoise_steps))

        print(f"[handler] Starting try-on | category={category} | steps={denoise_steps} | seed={seed}")

        # ---- Run inference ----
        result_image = run_tryon(
            person_image=person_image,
            garment_image=garment_image,
            garment_description=garment_description,
            category=category,
            denoise_steps=denoise_steps,
            seed=seed,
        )

        # ---- Encode result ----
        result_b64 = encode_pil_to_base64(result_image, fmt="PNG")

        print("[handler] Try-on complete")
        return {
            "status": "success",
            "result_image": result_b64,
        }

    except torch.cuda.OutOfMemoryError:
        # Special case: GPU OOM — surface a clear message rather than a crash
        torch.cuda.empty_cache()
        return {
            "status": "error",
            "error": (
                "CUDA out of memory. "
                "Try using a GPU with more VRAM (24 GB recommended), "
                "or reduce denoise_steps."
            ),
        }

    except Exception:
        # Catch everything else and return the traceback as a string so the
        # caller gets a meaningful error rather than a generic 500.
        tb = traceback.format_exc()
        print(f"[handler] ERROR:\n{tb}")
        return {
            "status": "error",
            "error": tb,
        }


# ---------------------------------------------------------------------------
# 6.  RunPod entrypoint
#     This line tells the RunPod SDK which function to call for each request.
# ---------------------------------------------------------------------------

runpod.serverless.start({"handler": handler})

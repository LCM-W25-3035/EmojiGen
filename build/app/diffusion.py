import io
import os
import torch
import base64
from PIL import Image
from peft import PeftModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Base model info
model_id = "sd-legacy/stable-diffusion-v1-5"
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the full pipeline once (without LoRA on UNet)
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
).to(device)

pipe.enable_attention_slicing()

# Disable NSFW filter
if pipe.safety_checker is not None:
    pipe.safety_checker = lambda images, clip_input, **kwargs: (images, [False] * len(images))

# Cache original UNet to clone later
base_unet = UNet2DConditionModel.from_pretrained(
    model_id,
    subfolder="unet",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
)

# Dict to store UNets with LoRA applied
lora_unets = {}

def diffusion(prompt: str, imgType: str) -> str:
    if imgType not in ["emoji", "sticker"]:
        raise ValueError("Invalid imgType. Choose 'emoji' or 'sticker'.")

    # Setup
    final_model_dir = os.path.join(current_dir, "diff_models", imgType)
    height, width = 256, 256
    guidance_scale = 7.5 if imgType == "emoji" else 8.5
    num_inference_steps = 50 if imgType == "emoji" else 60

    # Load LoRA-applied UNet only once
    if imgType not in lora_unets:
        print(f"Loading LoRA adapter for: {imgType}")
        unet_with_lora = PeftModel.from_pretrained(base_unet.to(device), final_model_dir)
        lora_unets[imgType] = unet_with_lora
    else:
        unet_with_lora = lora_unets[imgType]

    pipe.unet = unet_with_lora

    # Generate image
    with torch.no_grad():
        result = pipe(prompt=prompt,
                      num_inference_steps=num_inference_steps,
                      guidance_scale=guidance_scale,
                      height=height,
                      width=width)
        image = result.images[0]

    # Convert to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64

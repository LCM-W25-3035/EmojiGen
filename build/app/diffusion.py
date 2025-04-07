import io
import os
import torch
import base64
from PIL import Image
from rembg import remove
from peft import PeftModel, LoraConfig
from diffusers import StableDiffusionPipeline

def remove_background(image):
    """Removes background from an RGBA image using rembg."""
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    output = remove(img_bytes)
    return Image.open(io.BytesIO(output)).convert("RGBA")

def emoji_diffusion(prompt, imgType):
    # Device configuration (use 'cuda' if available, otherwise 'mps' or 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    # Load the base pipeline
    model_id = "sd-legacy/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    ).to(device)
    pipe.enable_attention_slicing()  # Memory-efficient optimization

    # Disable the safety checker to bypass NSFW filtering
    if pipe.safety_checker is not None:
        pipe.safety_checker = lambda images, clip_input, **kwargs: (images, [False] * len(images))

    # Q-LoRA configuration (must match your training configuration)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none"
    )

    # Determine model directory based on imgType
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if imgType == "emoji":
        final_model_dir = os.path.join(current_dir, "diff_models", "emoji")
        height, width = 256, 256
        guidance_scale = 7.5
        num_inference_steps = 50
    elif imgType == "sticker":
        final_model_dir = os.path.join(current_dir, "diff_models", "sticker")
        height, width = 256, 256
        guidance_scale = 8.5  # Stickers might need sharper definition
        num_inference_steps = 60
    else:
        raise ValueError("Invalid imgType. Choose 'emoji' or 'sticker'.")

    # Load the LoRA weights onto the UNet using PeftModel.from_pretrained
    pipe.unet = PeftModel.from_pretrained(pipe.unet, final_model_dir)

    # Generate the image
    with torch.no_grad():
        generated = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=height, width=width)
        image = generated.images[0]

    # Remove background for transparent output
    image = remove_background(image)

    # Save the generated image
    # Convert the image to a bytes buffer and encode as base64.
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64
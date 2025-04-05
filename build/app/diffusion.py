import io
import os
import torch
import base64
from PIL import Image
from peft import PeftModel, LoraConfig
from diffusers import StableDiffusionPipeline

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

    # Directory for the final trained LoRA weights
    current_dir = os.path.dirname(os.path.abspath(__file__))
    final_model_dir = os.path.join(current_dir, "diff_models", "emoji")

    # Load the LoRA weights onto the UNet using PeftModel.from_pretrained
    pipe.unet = PeftModel.from_pretrained(pipe.unet, final_model_dir)

    # Inference parameters
    num_inference_steps = 50
    guidance_scale = 7.5

    # Generate the image
    with torch.no_grad():
        generated = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=256, width=256)
        image = generated.images[0]

    # Save the generated image
    # Convert the image to a bytes buffer and encode as base64.
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64
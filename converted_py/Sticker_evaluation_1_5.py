import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel, LoraConfig
from PIL import Image
import os

# Set device to CUDA (GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the base Stable Diffusion pipeline
model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16  # Use float16 for faster inference on GPU
).to(device)

# Enable attention slicing to save memory (optional but recommended)
pipe.enable_attention_slicing()

# Disable NSFW safety checker (if needed)
if pipe.safety_checker is not None:
    pipe.safety_checker = lambda images, clip_input, **kwargs: (images, [False] * len(images))

# Load LoRA weights (make sure the directory is correct)
final_model_dir = "../evaluation/sticker_diffusion_qlora/checkpoint_epoch_25"
pipe.unet = PeftModel.from_pretrained(pipe.unet, final_model_dir)




# Inference parameters
prompt = "dolphin in water"
num_inference_steps = 50
guidance_scale = 7.5

# Generate the image
with torch.inference_mode():
    result = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=256,
        width=256
    )
    image = result.images[0]

# Save the image
output_path = "outputs/dolphin.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
image.save(output_path)
print(f"✅ Image saved at: {output_path}")

# Optionally display the image
image.show()





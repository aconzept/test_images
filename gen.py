# Renders one icon per entry of a JSON file {"name": "subject description"}
# into /out/<name>.png with SDXL base, skipping ones already there.
import json, os, sys, time, pathlib, torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

STYLE = ("a single 3D icon of {s}, cartoon, not a photo, glossy clay render, soft studio lighting, smooth rounded shapes, "
         "vibrant colors, centered single object, isolated on a plain soft warm cream background, "
         "minimal, Microsoft Fluent emoji style, high detail")
NEG = ("text, letters, watermark, logo, label, photo, photorealistic, realistic, noisy, grainy, "
       "multiple views, cluttered, frame, border, cropped, hands, people")

items = json.load(open(sys.argv[1]))
out = pathlib.Path('/out'); out.mkdir(exist_ok=True)
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 1

vae = AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix', torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0', vae=vae,
    torch_dtype=torch.float16, variant='fp16', use_safetensors=True)
# 8 GB is not enough for everything at once: keep only the active model on the GPU.
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()
# LIGHTNING=1: ByteDance's 8-step distilled LoRA, run without guidance.
lightning = os.environ.get('LIGHTNING') == '1'
if lightning:
    from diffusers import EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    pipe.load_lora_weights(hf_hub_download('ByteDance/SDXL-Lightning', 'sdxl_lightning_8step_lora.safetensors'))
    pipe.fuse_lora()
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing='trailing')

for name, subject in items.items():
    for seed in range(seeds):
        f = out / (f'{name}.png' if seeds == 1 else f'{name}.{seed}.png')
        if f.exists():
            continue
        t = time.time()
        img = pipe(STYLE.format(s=subject), negative_prompt=None if lightning else NEG, num_inference_steps=steps,
                   guidance_scale=0.0 if lightning else 7.0, width=1024, height=1024,
                   generator=torch.Generator('cuda').manual_seed(seed)).images[0]
        img.save(f)
        print(f'{f.name} {time.time() - t:.0f}s', flush=True)

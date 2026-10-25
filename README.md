# test_images

3D icon-style pictures for the 56 products of the yogarden test catalog,
two variants each (`products/<name>.0.png`, `products/<name>.1.png`),
1024×1024 PNG. All at a glance: [contact-sheet.png](contact-sheet.png).

Generated locally on a GTX 1080 with SDXL base 1.0, the
`madebyollin/sdxl-vae-fp16-fix` VAE and ByteDance's SDXL-Lightning 8-step
LoRA (about 50 s an image), in Docker:

    docker build -t yogarden-sdxl .
    docker run --rm --gpus all -e LIGHTNING=1 -v yogarden-sdxl-cache:/cache \
      -v "$PWD":/work:ro -v "$PWD/out":/out yogarden-sdxl python gen.py products.json 8 2

`products.json` holds each product's subject; `gen.py` wraps it in the
shared style prompt. Seeds are 0 and 1, so a run gives the same images.

Weak ones, worth a new prompt: Sujukh (neither looks like it), Butter.0,
Gata (looks like a medallion), Matnakash.0, Carrots, Cucumbers.1 and
Spinach.1 (tiled patterns).

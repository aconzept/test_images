# PyTorch 2.5 + CUDA 12.4 still ships kernels for Pascal (GTX 10xx, sm_61).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
RUN pip install --no-cache-dir diffusers==0.31.0 transformers==4.46.3 accelerate==1.1.1 safetensors==0.4.5 peft==0.13.2 pillow
ENV HF_HOME=/cache
WORKDIR /work

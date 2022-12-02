# python scripts/convert_stable_diffusion_checkpoint_to_onnx.py 
# --model_path="CompVis/stable-diffusion-v1-4" --output_path="./sd_onnx16" --fp16

# onnxruntime-gpu==1.12.0
import time
from diffusers import StableDiffusionOnnxPipeline

device = "cuda"
pipe = StableDiffusionOnnxPipeline.from_pretrained("./sd_onnx16", provider="CUDAExecutionProvider")
pipe = pipe.to(device)

prompt = "a photo of an astronaut riding a horse on mars"

start = time.time()
for i in range(10):
    image = pipe(prompt).images[0]
end = time.time()
print("time:", round(end-start, 3), "s")

image.save("astronaut_rides_horse.png")

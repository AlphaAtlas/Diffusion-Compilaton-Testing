from diffusers import StableDiffusionPipeline
import torch
import torch._dynamo
import tomesd

#/home/alpha/Storage/AIModels/Stable-diffusion/epicrealism_newAge
#runwayml/stable-diffusion-v1-5
#Create pipeline
#Change path to a remote model
pipe = StableDiffusionPipeline.from_pretrained("/home/alpha/Storage/AIModels/Stable-diffusion/epicrealism_newAge", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

#Typical diffusers pipeline optimizations
print("### Pre Optimization Benchmark:")
image = pipe("a photo of an astronaut riding a horse on mars").images[0]

pipe.unet.to(memory_format=torch.channels_last)
pipe.enable_vae_tiling()
pipe.enable_xformers_memory_efficient_attention()

print("### Post optimization benchmark:")
image = pipe("a photo of an astronaut riding a horse on mars").images[0]

tomesd.apply_patch(pipe, ratio=0.5)

image = pipe("a photo of an astronaut riding a horse on mars").images[0]

#Set compile
#torch._dynamo.config.suppress_errors = True

pipe.unet = torch.compile(pipe.unet, dynamic=True)

print("### torch.compile warmup:")
image = pipe("a photo of an astronaut riding a horse on mars").images[0]

print("torch.compile benchmark:")
image = pipe("a photo of an astronaut riding a horse on mars").images[0]





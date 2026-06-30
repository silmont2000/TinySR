"""
Minimal fast Hadamard debug: test forward pass with padding + hooks.
Usage: python test/test_fast_hadamard_debug.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from peft import LoraConfig
from diffusers import StableDiffusion3Pipeline
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from models.quant.layers import QuantLinearW4A4
from models.quant.inference import replace_quant_layers
from models.quant.hadamard import enable_rotation
from utils.util import load_lora_state_dict

os.environ["HF_HOME"] = "/data/disk2/xby/models"
os.environ["HF_HUB_CACHE"] = "/data/disk2/xby/models"

device = "cuda"
dtype = torch.float16

print("[1/5] Loading backbone + LoRA...")
transformer = TinySD3Transformer2DModel.from_pretrained(
    "checkpoint/tinybackbone/prune-12-merge-tinysr",
    subfolder="transformer", torch_dtype=dtype,
    low_cpu_mem_usage=False, ignore_mismatched_sizes=True,
)
lora_config = LoraConfig(r=64, lora_alpha=64, init_lora_weights='gaussian',
    target_modules=['to_k','to_q','to_v','to_out.0','proj','linear','linear_1','linear_2','net.2'])
transformer.add_adapter(lora_config); transformer.enable_adapters()
lora_sd = StableDiffusion3Pipeline.lora_state_dict("checkpoint/tinysr", weight_name="transformer.safetensors", cache_dir=os.environ["HF_HUB_CACHE"])
load_lora_state_dict(lora_sd, transformer)
transformer = transformer.merge_and_unload().to(device).eval()

print("[2/5] Replacing with QuantLinearW4A4...")
replace_quant_layers(transformer, "dit_full", quant_config=None,
    w_bits=4, a_bits=4, svdq_rank=32, svdq_smooth_alpha=1.0,
    svdq_iterations=0, act_group_size=64, weight_group_size=-1)

print("[3/5] Enabling fast Hadamard rotation...")
enable_rotation(transformer, mode="fast_hadamard")

print("[4/5] Creating test input...")
timesteps = torch.tensor([1000.0], device=device, dtype=dtype)
pooled = torch.load("dataset/default/pool_embeds.pt", map_location=device).to(dtype=dtype)
x = torch.randn(1, 16, 64, 64, device=device, dtype=dtype)

start = time.time()
print("  calling transformer...")
out = transformer(hidden_states=x, timestep=timesteps, pooled_projections=pooled, return_dict=False)[0]
elapsed = time.time() - start

print(f"[5/5] DONE in {elapsed:.2f}s, output shape: {out.shape}")

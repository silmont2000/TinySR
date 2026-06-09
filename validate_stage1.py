# fmt:off
'''
# --------------------------------------------------------------------------------
#    script modified from  (https://github.com/Microtreei/TSD-SR)
# --------------------------------------------------------------------------------
'''
import glob
import time
import os
import sys
import json as _json
sys.path.append(".")
import argparse
from PIL import Image
import torch
import torch.nn.functional as tfF
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from peft import LoraConfig
from diffusers import (
    StableDiffusion3Pipeline,
)
from models.tinysr.stage1_defaults import (
    CKPT, DEFAULT_PYRAMID_CONFIG, LORA_R, SMOKE_RANK_PATTERN, make_lora_config,
    DEFAULT_TIMESTEP,
)
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.tinysr.pyramid_config import PyramidArchConfig
from models.tinysr.pyramid_model import TinyPyramidSD3Transformer2DModel
from models.vae.autoencoder_tiny  import  AutoencoderTiny
from models.vae.autoencoder_kl  import  AutoencoderKL

from models.quant.tiler import tile_sample

from utils.vaehook import _init_tiled_vae
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from utils.util import load_lora_state_dict_warn

from torchinfo import summary

is_print=True

def print_module_details(module, file, indent=0, name="model"):
    # print(module, file=file)
    prefix = "  " * indent
    print(f"{prefix}{name}: {module.__class__.__name__}", file=file)
    for param_name, param in module.named_parameters(recurse=False):
        print(f"{prefix}  [P] {param_name}: {param.shape}", file=file)
    for buffer_name, buffer in module.named_buffers(recurse=False):
        print(f"{prefix}  [B] {buffer_name}: {buffer.shape}", file=file)
    for child_name, child in module.named_children():
        print_module_details(child, file, indent + 1, child_name)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="path/to/your/model", help='path to the pretrained sd3')
    parser.add_argument("--vae_path", type=str, default="path/to/your/vae", help='path to tsd-sr lora weights')
    parser.add_argument("--lora_dir", type=str, default="path/to/your/lora", help='path to tsd-sr lora weights')
    parser.add_argument("--cache_dir", type=str, default="/data/disk2/xby/models", help='cache directory for downloading models')
    parser.add_argument("--embedding_dir", type=str, default="dataset/default/", help='path to prompt embeddings')
    parser.add_argument("--output_dir", '-o', type=str, default="outputs/tinysr/", help='path to save results')
    parser.add_argument('--input_dir', '-i', type=str, default="path/to/your/input", help='path to the input image')

    parser.add_argument("--rank", type=int, default=64, help='rank for transformer')
    parser.add_argument("--rank_vae", type=int, default=64, help='rank for vae')

    parser.add_argument("--is_use_tile", type=bool, default=False, help='whether to use tiled vae')
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224, help='tiled size for tiled vae decoder') 
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024, help='tiled size for tiled vae encoder') 
    parser.add_argument("--latent_tiled_size", type=int, default=64, help='tiled size for transformer latent')
    parser.add_argument("--latent_tiled_overlap", type=int, default=8, help='tiled overlap for transformer latent')

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upscale", type=int, default=4, help='upscale factor')
    parser.add_argument("--process_size", type=int, default=512, help='process size for images')
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16")
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain', help='color alignment method')
    parser.add_argument("--eval_latent_dir", type=str, default="dataset/smoke/latent_stu",
                        help="Directory of teacher latents (.pt) for eval mode")
    parser.add_argument("--eval_size", type=int, default=128,
                        help="Resize images to this size before VAE encode in eval mode")
    return parser.parse_args()



tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

def main(args, pixel_values):
    with torch.no_grad():
        # Preprocess the input image
        # pixel_values = torch.nn.functional.interpolate(pixel_values, size=size, mode='bicubic', align_corners=False)
        pixel_values = pixel_values * 2 - 1
        pixel_values = pixel_values.to(args.device, dtype=weight_dtype)

        # Encode the input image
        model_input = vae.encode(pixel_values).latents * vae.config.scaling_factor
        # model_input = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
        model_input = model_input.to(args.device, dtype=weight_dtype)

        # Predict
        output, pre_last, _ = transformer(
            hidden_states=model_input, timestep=timesteps,
            pooled_projections=pooled_prompt_embeds, return_dict=False,
        )
        last_grid_hw = transformer.pyramid_config.p_states[-1].grid_hw
        if pyramid_loss_type == "a":
            scale_factor = 64 // pc.sample_size
            # scale_factor = last_grid_hw // transformer.pyramid_config.p_states[0].grid_hw
            input_up = tfF.interpolate(model_input, scale_factor=scale_factor,
                                        mode='bilinear', align_corners=False)
            denoised = input_up - output
        else:
            latent_before = transformer._tokens_to_latent(pre_last, last_grid_hw)
            denoised = latent_before - output
        
        # denoised = denoised.to(args.device, dtype=weight_dtype)

        # Decode the output
        image = vae.decode(denoised / vae.config.scaling_factor, return_dict=False)[0].squeeze(0).clamp(-1,1)

        return image,denoised

if __name__ == "__main__":
    args = parse_args()

    # Set huggingface cache directory
    os.environ['HF_HOME'] = args.cache_dir
    os.environ['HF_HUB_CACHE'] = args.cache_dir

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    # ── Pyramid config ──
    has_lora = args.lora_dir and os.path.isdir(args.lora_dir)
    if has_lora:
        pc_path = os.path.join(args.lora_dir, "pyramid_config.json")
        if os.path.exists(pc_path):
            with open(pc_path) as f:
                pc = PyramidArchConfig.from_dict(_json.load(f))
        else:
            pc = DEFAULT_PYRAMID_CONFIG
    else:
        pc = DEFAULT_PYRAMID_CONFIG
    print(f"  pyramid: sample_size={pc.sample_size}, {[(s.num_blocks,s.dim,s.grid_hw) for s in pc.p_states]}")
    transformer = TinyPyramidSD3Transformer2DModel.from_flat_pretrained(
        args.pretrained_model_name_or_path, pyramid_config=pc,
        subfolder="transformer", torch_dtype=weight_dtype,
    )
    vae = AutoencoderTiny.from_pretrained(args.vae_path, torch_dtype=weight_dtype, cache_dir=args.cache_dir)
    vae_decode = AutoencoderKL.from_pretrained("/data/disk2/xby/sd3-medium", subfolder="vae").to("cuda", weight_dtype)
    # vae = AutoencoderKL.from_pretrained("/data/disk2/xby/sd3-medium", subfolder="vae").to("cuda", weight_dtype)

    if args.is_use_tile:
        _init_tiled_vae(vae, encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    if has_lora:
        rp_path = os.path.join(args.lora_dir, "rank_pattern.json")
        if os.path.exists(rp_path):
            with open(rp_path) as f:
                rank_pattern = _json.load(f)
        else:
            rank_pattern = None
        transformer_lora_config = make_lora_config(rank_pattern=rank_pattern)
        
        transformer.add_adapter(transformer_lora_config)
        transformer.enable_adapters()

        transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(args.lora_dir, weight_name="model.safetensors", cache_dir=args.cache_dir)
        transformer_lora_state_dict = dict(transformer_lora_state_dict)
        load_lora_state_dict_warn(transformer_lora_state_dict, transformer)
        if transformer_lora_state_dict:
            print(f"  Warning: {len(transformer_lora_state_dict)} LoRA keys not loaded:")
            sample_keys = list(transformer_lora_state_dict.keys())[:3]
            for k in sample_keys:
                print(f"    {k}")

    # ── Pyramid: load loss_type for denoising ──
    pyramid_loss_type = "a"
    if has_lora:
        tc_path = os.path.join(args.lora_dir, "train_config.json")
        if os.path.exists(tc_path):
            with open(tc_path) as f:
                pyramid_loss_type = _json.load(f).get("loss_type", "a")
    print(f"  pyramid loss_type={pyramid_loss_type}")
    
    vae = vae.to(args.device, dtype=weight_dtype)
    transformer = transformer.to(args.device, dtype=weight_dtype)

    param_cnt = sum(p.numel() for p in (transformer.p_states).parameters())
    print("#Param.", param_cnt/1e6, "M")

    # Sample timestep for each image
    timesteps = torch.tensor([1000.], device=args.device, dtype=weight_dtype)

    # Load the prompt embeddings
    prompt_default = "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extrememeticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations."
    pooled_prompt_embeds = torch.load(os.path.join(args.embedding_dir, "pool_embeds.pt"), map_location=args.device).to(dtype=weight_dtype)

    # Get the image names
    if os.path.isdir(args.input_dir):
        image_names = sorted(glob.glob(f'{args.input_dir}/*.png'))
    else:
        image_names = [args.input_dir]

    datalen = len(image_names)
    print("image_num", datalen)
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    total_time = 0.0
    mem_records = []

    tqdm_files = tqdm(image_names, desc="Eval")
    total_loss = 0.0
    total_count = 0

    structure_file = os.path.join(args.output_dir, "parimaid_model_structure.txt")
    with open(structure_file, "w") as f:
        print("========== VAE detailed ==========", file=f)
        print_module_details(vae, file=f, name="vae")
        print("\n========== Transformer detailed ==========", file=f)
        print_module_details(transformer, file=f, name="transformer")
        print(f"\n========== saved module detail to{structure_file} ==========")


    # Warmup: 5 forward passes on first image for GPU warmup
    torch.cuda.empty_cache()
    if image_names:
        first_lr = Image.open(image_names[0]).convert('RGB')
        first_lr = first_lr.resize((args.eval_size, args.eval_size), Image.BICUBIC)
        warmup_pixels = tensor_transforms(first_lr).unsqueeze(0).to(args.device, dtype=weight_dtype)
        transformer.eval()
        for _ in range(5):
            main(args, warmup_pixels)
        print("Warmup done (5 iters).")


    for image_name in tqdm_files:
        stem = os.path.splitext(os.path.basename(image_name))[0]
        latent_path = os.path.join(args.eval_latent_dir, f"{stem}.pt")
        has_latent = os.path.exists(latent_path)

        lr = Image.open(image_name).convert('RGB')
        lr = lr.resize((args.eval_size, args.eval_size), Image.BICUBIC)
        ori_width, ori_height = lr.size
        upscale = args.upscale
        process_size = args.process_size

        resize_flag = False

        lr_scale = lr.resize((int(ori_width*args.upscale), int(ori_height*args.upscale)))
        pixel_values = tensor_transforms(lr).unsqueeze(0).to(args.device, dtype=weight_dtype)
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        image,denoised = main(args, pixel_values)
        torch.cuda.synchronize()
        end_time = time.time()
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        total_time += (end_time - start_time)
        mem_records.append(peak_mem)

        if has_latent:
            tg = torch.load(latent_path, map_location='cpu')
            if tg.dim() == 3:
                tg = tg.unsqueeze(0)
            loss = tfF.l1_loss(denoised.float().cpu(), tg.float())
            total_loss += loss.item()
            total_count += 1
            tqdm_files.set_postfix(loss=f"{total_loss/total_count:.6f}")
        
        image_pil_image = transforms.ToPILImage()(image.cpu() / 2 + 0.5)

        if args.align_method == 'adain':
            image_pil_image = adain_color_fix(target=image_pil_image, source=lr)
        elif args.align_method == 'wavelet':
            image_pil_image = wavelet_color_fix(target=image_pil_image, source=lr_scale)
        else:
            pass

        image_pil_image.save(os.path.join(args.output_dir, os.path.basename(image_name)))
        torch.cuda.empty_cache()
    param_cnt = sum(p.numel() for p in transformer.p_states.parameters())
    mem_arr = np.array(mem_records)
    print("#Param.", param_cnt/1e6, "M")
    print(f"Average time: {total_time / datalen:.4f} sec/image")
    if len(mem_arr) > 0:
        print(f"Peak mem  avg: {np.mean(mem_arr):.0f} MB")
        print(f"Peak mem  max: {np.max(mem_arr):.0f} MB")




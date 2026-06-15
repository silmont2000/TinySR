import glob
import time
import os
import sys
sys.path.append(".")
import argparse
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
from peft import LoraConfig
from diffusers import (
    StableDiffusion3Pipeline,
)
from models.final_sd3.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny  import  AutoencoderTiny

from utils.vaehook import _init_tiled_vae
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from utils.util import load_lora_state_dict, load_lora_state_dict_warn
import json as _json
import torch.nn.functional as tfF
from models.tinysr.stage1_defaults import DEFAULT_PYRAMID_CONFIG, make_lora_config
from models.tinysr.pyramid_config import PyramidArchConfig
from models.tinysr.pyramid_model import TinyPyramidSD3Transformer2DModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="checkpoint/tinybackbone/prune-12-merge-tinysr", help='path to the pretrained sd3')
    parser.add_argument("--lora_dir", type=str, default="checkpoint/tinysr", help='path to tsd-sr lora weights')
    # parser.add_argument("--vae_path", type=str, default="checkpoint/taesd/train-proj/v2/checkpoint-60500/vae.safetensors", help='path to tsd-sr lora weights')
    parser.add_argument("--vae_path", type=str, default="checkpoint/vae/separable", help='path to tsd-sr lora weights')
    
    parser.add_argument("--embedding_dir", type=str, default="dataset/default/", help='path to prompt embeddings')
    parser.add_argument("--output_dir", '-o', type=str, default="outputs/final/tinysr/dreal", help='path to save results')
    # parser.add_argument('--input_dir', '-i', type=str, default="/data/wzh/StableSR_testsets/DrealSRVal_crop128/test_LR/", help='path to the input image')
    
    # parser.add_argument('--input_dir', '-i', type=str, default="/data/wzh/StableSR_testsets/DIV2K_V2_Val_500/test_LR/", help='path to the input image')
    # parser.add_argument('--input_dir', '-i', type=str, default="/data/wzh/StableSR_testsets/DIV2K_V2_val/test_LR/", help='path to the input image')
    parser.add_argument('--input_dir', '-i', type=str, default="dataset/StableSR_testsets/DrealSRVal_crop128/test_LR", help='path to the input image')
    # parser.add_argument('--input_dir', '-i', type=str, default="/data/wzh/StableSR_testsets/RealSRVal_crop128/test_LR/", help='path to the input image')

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
    parser.add_argument("--pyramid", action="store_true", help="use pyramid model")
        
    return parser.parse_args()

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

def main(args, pixel_values, size):
    with torch.no_grad():
        # Preprocess the input image
        pixel_values = torch.nn.functional.interpolate(pixel_values, size=size, mode='bicubic', align_corners=False)
        pixel_values = pixel_values * 2 - 1
        pixel_values = pixel_values.to(args.device, dtype=weight_dtype).clamp(-1,1)
        
        # Encode the input image
        model_input = vae.encode(pixel_values).latents * vae.config.scaling_factor
        model_input = model_input.to(args.device, dtype=weight_dtype)

        if args.pyramid:
            output, pre_last, _ = transformer(
                hidden_states=model_input, timestep=timesteps,
                pooled_projections=pooled_prompt_embeds, return_dict=False,
            )
            last_grid_hw = transformer.pyramid_config.p_states[-1].grid_hw
            if pyramid_loss_type == "a":
                scale_factor = 64 // pc.sample_size
                input_up = tfF.interpolate(model_input, scale_factor=scale_factor,
                                            mode='bilinear', align_corners=False)
                denoised = input_up - output
            else:
                latent_before = transformer._tokens_to_latent(pre_last, last_grid_hw)
                denoised = latent_before - output
        else:
            model_pred = transformer(
                hidden_states=model_input,
                timestep=timesteps,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
            denoised = model_input - model_pred
        
        # Decode the output
        image = vae.decode(denoised / vae.config.scaling_factor, return_dict=False)[0].squeeze(0).clamp(-1,1)
        
        return image

if __name__ == "__main__":
    args = parse_args()
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
        
    # Load the pretrained models
    pyramid_loss_type = "a"
    pc = None
    if args.pyramid:
        pc_path = os.path.join(args.lora_dir, "pyramid_config.json")
        if os.path.exists(pc_path):
            with open(pc_path) as f:
                pc = PyramidArchConfig.from_dict(_json.load(f))
        else:
            pc = DEFAULT_PYRAMID_CONFIG
        print(f"  pyramid: sample_size={pc.sample_size}, {[(s.num_blocks,s.dim,s.grid_hw) for s in pc.p_states]}")
        mult_config_path = None
        candidate = os.path.join(args.lora_dir, "mult_config.json")
        if os.path.isfile(candidate):
            mult_config_path = candidate
            print(f"  mult_config loaded from {mult_config_path}")
        transformer = TinyPyramidSD3Transformer2DModel.from_flat_pretrained(
            args.pretrained_model_name_or_path, pyramid_config=pc,
            subfolder="transformer", torch_dtype=weight_dtype,
            mult_config_path=mult_config_path,
        )
    else:
        transformer = TinySD3Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="transformer", 
                                                torch_dtype=weight_dtype, low_cpu_mem_usage=False, ignore_mismatched_sizes=True,attn_implementation="flash_attention_2")
    
    vae = AutoencoderTiny.from_pretrained(args.vae_path, torch_dtype=weight_dtype)
    
    # if args.pyramid:
        # rp_path = os.path.join(args.lora_dir, "rank_pattern.json")
        # if os.path.exists(rp_path):
        #     with open(rp_path) as f:
        #         rank_pattern = _json.load(f)
        # else:
        #     rank_pattern = None
        # transformer_lora_config = make_lora_config(rank_pattern=rank_pattern)
        # transformer.add_adapter(transformer_lora_config)
        # transformer.enable_adapters()
        # transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(args.lora_dir, weight_name="model.safetensors")
        # load_lora_state_dict_warn(transformer_lora_state_dict, transformer)
        # tc_path = os.path.join(args.lora_dir, "train_config.json")
        # if os.path.exists(tc_path):
        #     with open(tc_path) as f:
        #         pyramid_loss_type = _json.load(f).get("loss_type", "a")
    
    print(transformer)
    if args.is_use_tile:
        _init_tiled_vae(vae, encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    vae = vae.to(args.device, dtype=weight_dtype)
    transformer = transformer.to(args.device, dtype=weight_dtype)
    transformer.eval()

    # Sample timestep for each image
    timesteps = torch.tensor([1000.], device=args.device, dtype=weight_dtype)

    # Load the prompt embeddings
    prompt_default = "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extrememeticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations."
    prompt_embeds = torch.load(os.path.join(args.embedding_dir, "prompt_embeds.pt"), map_location=args.device).to(dtype=weight_dtype)
    pooled_prompt_embeds = torch.load(os.path.join(args.embedding_dir, "pool_embeds.pt"), map_location=args.device).to(dtype=weight_dtype)
   
    total_time = 0.
    batch_size = 10
    inference_iterations = 100

    pixel_values = torch.randn(inference_iterations, batch_size ,3,128, 128, dtype=weight_dtype, device=args.device)
    for pixel_value in pixel_values:
        image = main(args, pixel_value, (args.process_size, args.process_size))
    
    for pixel_value in tqdm(pixel_values, desc="Inference"):
        start_time = time.time()
        
        image = main(args, pixel_value, (args.process_size, args.process_size))
        
        torch.cuda.synchronize()
        end_time = time.time()
        total_time += (end_time - start_time)
        # torch.cuda.empty_cache() # 不加更快
        
    print(transformer)
    print(f"Average time: {total_time / inference_iterations / batch_size}")
            



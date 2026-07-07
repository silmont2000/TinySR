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
from utils.util import load_lora_state_dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="checkpoint/tinyfusion/tinymerge/prune-12-merge-tinysr", help='path to the pretrained sd3')
    parser.add_argument("--lora_dir", type=str, default="checkpoint/train-tinysr/gan_tiny/v1/checkpoint-40500", help='path to tsd-sr lora weights')
    # parser.add_argument("--vae_path", type=str, default="checkpoint/taesd/train-proj/v2/checkpoint-60500/vae.safetensors", help='path to tsd-sr lora weights')
    parser.add_argument("--vae_path", type=str, default="checkpoint/taesd/train-sparse/v1/checkpoint-24500/vae.safetensors", help='path to tsd-sr lora weights')
    
    parser.add_argument("--embedding_dir", type=str, default="dataset/default/", help='path to prompt embeddings')
    parser.add_argument("--output_dir", '-o', type=str, default="outputs/final/tinysr/dreal", help='path to save results')
    # parser.add_argument('--input_dir', '-i', type=str, default="/data/wzh/StableSR_testsets/DrealSRVal_crop128/test_LR/", help='path to the input image')
    
    # parser.add_argument('--input_dir', '-i', type=str, default="/data/wzh/StableSR_testsets/DIV2K_V2_Val_500/test_LR/", help='path to the input image')
    # parser.add_argument('--input_dir', '-i', type=str, default="/data/wzh/StableSR_testsets/DIV2K_V2_val/test_LR/", help='path to the input image')
    parser.add_argument('--input_dir', '-i', type=str, default="/data/wzh/StableSR_testsets/DrealSRVal_crop128/test_LR/", help='path to the input image')
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

        # Predict
        model_pred =  transformer(
                    hidden_states=model_input,
                    timestep=timesteps,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
        latent_stu = model_input - model_pred
        
        # Decode the output
        image = vae.decode(latent_stu / vae.config.scaling_factor, return_dict=False)[0].squeeze(0).clamp(-1,1)
        
        return image

if __name__ == "__main__":
    args = parse_args()
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
        
    # Load the pretrained models
    transformer = TinySD3Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="transformer", 
                                            torch_dtype=weight_dtype, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
    
    vae = AutoencoderTiny.from_pretrained(args.vae_path, torch_dtype=weight_dtype)
    
    print(transformer)
    if args.is_use_tile:
        _init_tiled_vae(vae, encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    vae = vae.to(args.device, dtype=weight_dtype)
    transformer = transformer.to(args.device, dtype=weight_dtype)

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
        image = main(args, pixel_value, (512, 512))
    
    for pixel_value in tqdm(pixel_values, desc="Inference"):
        start_time = time.time()
        
        image = main(args, pixel_value, (512, 512))
        
        torch.cuda.synchronize()
        end_time = time.time()
        total_time += (end_time - start_time)
        # torch.cuda.empty_cache() # 不加更快
        
    print(transformer)
    print(f"Average time: {total_time / inference_iterations / batch_size}")
            



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
    SD3Transformer2DModel,
)
# from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny  import  AutoencoderTiny
from models.vae.autoencoder_kl import AutoencoderKL
from data import Real_ESRGAN_Dataset

FLICKR2K_PATH = "flicker data path"  
DIV2K_PATH = "div2k path"     
LSDIR20K_PATH = "lsdir path"     
FFHQ10K_PATH = "ffhq path"        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="your/teacher", help='path to the pretrained sd3')

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

def main(args, batch):
    with torch.no_grad():
        lr_values = batch["lr_img"].to(args.device, dtype=weight_dtype).unsqueeze(0)
        prompt_embeds = batch["prompt_embeds_input"].to(args.device, dtype=weight_dtype)
        pooled_prompt_embeds = batch["pooled_prompt_embeds_input"].to(args.device, dtype=weight_dtype)
        img_name = batch["img_name"]

        # Encode the input image
        model_input = vae.encode(lr_values).latent_dist.sample() * vae.config.scaling_factor
        model_input = model_input.to(args.device, dtype=weight_dtype)

        # Predict
        model_pred =  transformer(
                    hidden_states=model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
        
        latent_stu = model_input - model_pred
        
        latent_stu_dir = img_name.replace(".png", ".pt").replace("sr_bicubic", "latent_stu")
        vae_stu_dir = img_name.replace(".png", ".pt").replace("sr_bicubic", "vae_stu")
        # if not os.path.exists(os.path.dirname(latent_stu_dir)):
        #     os.makedirs(os.path.dirname(latent_stu_dir))
        # if not os.path.exists(os.path.dirname(vae_stu_dir)):
        #     os.makedirs(os.path.dirname(vae_stu_dir))
        # torch.save(latent_stu.cpu().float(), latent_stu_dir)
        # torch.save(latent_stu.cpu().float(), vae_stu_dir)
        
        
def main2(args, batch):
    with torch.no_grad():
        lr_values = batch["lr_img"].to(args.device, dtype=weight_dtype).unsqueeze(0)
        img_name = batch["img_name"]

        # Encode the input image
        model_input = vae.encode(lr_values).latent_dist.sample() * vae.config.scaling_factor

        
        vae_stu_dir = img_name.replace(".png", ".pt").replace("sr_bicubic", "vae_stu")
        # if not os.path.exists(os.path.dirname(vae_stu_dir)):
        #     os.makedirs(os.path.dirname(vae_stu_dir))
        # torch.save(model_input.cpu().float(), vae_stu_dir)

def run_encode():
    # data_dir = [FLICKR2K_PATH ]
    dataset = Real_ESRGAN_Dataset(device="cpu",process_size=512)
    for data in tqdm(dataset):
        main(args, data)


def run_vae():
    # data_dir = [FLICKR2K_PATH ]
    dataset = Real_ESRGAN_Dataset(device="cpu",process_size=512)
    for data in tqdm(dataset):
        main2(args, data)

if __name__ == "__main__":
    args = parse_args()
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
        
    # Load the pretrained models
    transformer = SD3Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="transformer", torch_dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype)
    
    vae = vae.to(args.device, dtype=weight_dtype)
    transformer = transformer.to(args.device, dtype=weight_dtype)

    # Sample timestep for each image
    timesteps = torch.tensor([1000.], device=args.device, dtype=weight_dtype)

    run_vae()
    torch.cuda.empty_cache()
        
            



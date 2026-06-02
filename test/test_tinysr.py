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
sys.path.append(".")
import argparse
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from peft import LoraConfig
from diffusers import (
    StableDiffusion3Pipeline,
)
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny  import  AutoencoderTiny
from models.vae.autoencoder_kl  import  AutoencoderKL
# from diffusers import AutoencoderKL

from models.quant.tiler import tile_sample

from utils.vaehook import _init_tiled_vae
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from utils.util import load_lora_state_dict

from torchinfo import summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="path/to/your/model", help='path to the pretrained sd3')
    parser.add_argument("--vae_path", type=str, default="path/to/your/vae", help='path to tsd-sr lora weights')
    parser.add_argument("--lora_dir", type=str, default=None, help='path to tsd-sr lora weights (omit for no-LoRA baseline)')
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
    return parser.parse_args()



tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

def main(args, pixel_values, size):
    with torch.no_grad():
        # Preprocess the input image
        pixel_values = torch.nn.functional.interpolate(pixel_values, size=size, mode='bicubic', align_corners=False)
        pixel_values = pixel_values * 2 - 1
        pixel_values = pixel_values.to(args.device, dtype=weight_dtype)

        # Encode the input image
        model_input = vae.encode(pixel_values).latents * vae.config.scaling_factor
        # model_input = vae_decode.encode(pixel_values).latents_dist.sample() * vae.config.scaling_factor
        model_input = model_input.to(args.device, dtype=weight_dtype)

        # Predict
        model_pred = tile_sample(model_input, transformer, timesteps, pooled_prompt_embeds, weight_dtype,
                                 latent_tiled_size=args.latent_tiled_size, latent_tiled_overlap=args.latent_tiled_overlap)

        latent_stu = model_input - model_pred

        # Decode the output
        image = vae_decode.decode(latent_stu / vae.config.scaling_factor, return_dict=False)[0].squeeze(0).clamp(-1,1)

        return image

if __name__ == "__main__":
    args = parse_args()

    # Set huggingface cache directory
    os.environ['HF_HOME'] = args.cache_dir
    os.environ['HF_HUB_CACHE'] = args.cache_dir

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    # Load the pretrained models
    transformer = TinySD3Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="transformer", 
                                            torch_dtype=weight_dtype, low_cpu_mem_usage=False, ignore_mismatched_sizes=True, cache_dir=args.cache_dir)
    vae = AutoencoderTiny.from_pretrained(args.vae_path, torch_dtype=weight_dtype, cache_dir=args.cache_dir)
    # vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=weight_dtype, cache_dir=args.cache_dir)
    vae_decode = AutoencoderKL.from_pretrained("/data/disk2/xby/sd3-medium", subfolder="vae").to("cuda", weight_dtype)

    if args.is_use_tile:
        _init_tiled_vae(vae, encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    if args.lora_dir:
        transformer_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0","proj","linear", "linear_1", "linear_2", "net.2"],
        )
        transformer.add_adapter(transformer_lora_config)
        transformer.enable_adapters()

        transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(args.lora_dir, weight_name="transformer.safetensors", cache_dir=args.cache_dir)
        load_lora_state_dict(transformer_lora_state_dict, transformer)

    vae = vae.to(args.device, dtype=weight_dtype)
    transformer = transformer.to(args.device, dtype=weight_dtype)

    param_cnt = sum(p.numel() for p in transformer.transformer_blocks.parameters() )
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

    torch.cuda.empty_cache()
    if (image_names):
        lr = Image.open(image_names[0]).convert('RGB')
        ori_width, ori_height = lr.size
        upscale = args.upscale
        process_size = args.process_size

        # Resize the image if it is not valid
        resize_flag = False
        if ori_width < process_size // upscale or ori_height < process_size // upscale:
            scale = (process_size // upscale) / min(ori_width, ori_height)
            new_width, new_height = int(scale*ori_width), int(scale*ori_height)
            resize_flag = True
        else:
            new_width, new_height = ori_width, ori_height
        new_width, new_height = upscale*new_width, upscale*new_height
        if new_width % 8 or new_height % 8:
            resize_flag = True
            new_width = new_width - new_width % 8
            new_height = new_height - new_height % 8

        lr_scale = lr.resize((int(ori_width*args.upscale), int(ori_height*args.upscale)))
        pixel_values = tensor_transforms(lr).unsqueeze(0).to(args.device, dtype=weight_dtype)
        for i in range(5):
            main(args, pixel_values, (new_height, new_width))

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    total_wall_time = 0.0
    total_gpu_time = 0.0
    mem_records = []
    for image_name in tqdm(image_names):
        lr = Image.open(image_name).convert('RGB')
        ori_width, ori_height = lr.size
        upscale = args.upscale
        process_size = args.process_size

        # Resize the image if it is not valid
        resize_flag = False
        if ori_width < process_size // upscale or ori_height < process_size // upscale:
            scale = (process_size // upscale) / min(ori_width, ori_height)
            new_width, new_height = int(scale*ori_width), int(scale*ori_height)
            resize_flag = True
        else:
            new_width, new_height = ori_width, ori_height
        new_width, new_height = upscale*new_width, upscale*new_height
        if new_width % 8 or new_height % 8:
            resize_flag = True
            new_width = new_width - new_width % 8
            new_height = new_height - new_height % 8

        lr_scale = lr.resize((int(ori_width*args.upscale), int(ori_height*args.upscale)))
        pixel_values = tensor_transforms(lr).unsqueeze(0).to(args.device, dtype=weight_dtype)

        torch.cuda.reset_peak_memory_stats()
        wall_start = time.time()
        start_ev.record()
        image = main(args, pixel_values, (new_height, new_width))
        end_ev.record()
        torch.cuda.synchronize()
        wall_end = time.time()

        gpu_time = start_ev.elapsed_time(end_ev) / 1000
        wall_time = wall_end - wall_start
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        image_pil_image = transforms.ToPILImage()(image.cpu() / 2 + 0.5)
        total_gpu_time += gpu_time
        total_wall_time += wall_time
        mem_records.append(peak_mem)
        if resize_flag:
            image_pil_image = image_pil_image.resize((int(ori_width*args.upscale), int(ori_height*args.upscale)))

        if args.align_method == 'adain':
            image_pil_image = adain_color_fix(target=image_pil_image, source=lr)
        elif args.align_method == 'wavelet':
            image_pil_image = wavelet_color_fix(target=image_pil_image, source=lr_scale)
        else:
            pass

        image_pil_image.save(os.path.join(args.output_dir, os.path.basename(image_name)))
    torch.cuda.empty_cache()
    param_cnt = sum(p.numel() for p in transformer.transformer_blocks.parameters() )
    mem_arr = np.array(mem_records)
    print("#Param.", param_cnt/1e6, "M")
    print(f"Average GPU  time: {total_gpu_time / datalen:.4f} sec/image")
    print(f"Average Wall time: {total_wall_time / datalen:.4f} sec/image")
    print(f"Peak mem  avg: {np.mean(mem_arr):.0f} MB")
    print(f"Peak mem  max: {np.max(mem_arr):.0f} MB")




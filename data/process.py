'''
# --------------------------------------------------------------------------------
#    script modified from  (https://github.com/Microtreei/TSD-SR)
# --------------------------------------------------------------------------------
'''
import os
import sys
sys.path.append(os.getcwd())
import glob
import argparse
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModelWithProjection, T5TokenizerFast
from diffusers import AutoencoderKL
from tqdm import tqdm
# from ram.models.ram_lora import ram
# from ram import inference_ram as inference
import pdb
import re
from torchvision import transforms
FLICKR2K_PATH = "/data/disk1/dlw/datasets/CAD_V100_20260209_backup/datasets/Flickr2K/Flickr2K_LRx4_Real-ESRGAN_Seesr_v2"
DIV2K_PATH = "/data/disk1/dlw/datasets/CAD_V100_20260209_backup/datasets/DIV2K/DIV2K_train_LRx4_Real-ESRGAN_Seesr_v2"
LSDIR20K_PATH = "lsdir path"     
FFHQ10K_PATH = "/data/disk1/dlw/datasets/CAD_V100_20260209_backup/datasets/FFHQ/FFHQ10K_LRx4_Real-ESRGAN_Seesr_v2"       

def merge_data(data_path, hr_name="gt", lr_name="sr_bicubic"):
    hr_data_file_path = []
    lr_data_file_path = []
    for data_dir in data_path:
        if not os.path.exists(data_dir):
            return
        hr_data_dir = os.path.join(data_dir, hr_name)
        lr_data_dir = os.path.join(data_dir, lr_name)
        
        hr_file = os.listdir(hr_data_dir)
        hr_file.sort(key=lambda x: int(x.split(".")[0]))
        
        lr_file = os.listdir(lr_data_dir)
        lr_file.sort(key=lambda x: int(x.split(".")[0]))
        
        sub_hr_data_dir = [os.path.join(hr_data_dir , file) for file in hr_file]
        sub_lr_data_dir = [os.path.join(lr_data_dir , file) for file in lr_file]
        
        hr_data_file_path = hr_data_file_path + sub_hr_data_dir
        lr_data_file_path = lr_data_file_path + sub_lr_data_dir
    
    return hr_data_file_path, lr_data_file_path

def merge_val_data(data_path, lr_name="test_LR"):
    lr_data_file_path = []
    for data_dir in data_path:
        if not os.path.exists(data_dir):
            return
        lr_data_dir = os.path.join(data_dir, lr_name)
        lr_file = os.listdir(lr_data_dir)
        
        sub_lr_data_dir = [os.path.join(lr_data_dir , file) for file in lr_file]
        
        lr_data_file_path = lr_data_file_path + sub_lr_data_dir
    
    return lr_data_file_path

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=256,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds

def import_text_encoder(pretrained_model_name_or_path = "defualt", device=None): 
    tokenizer_one = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_2",
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_3",
    )

    # import correct text encoder classes
    text_encoder_one = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_2").to(device)
    text_encoder_three = T5EncoderModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_3").to(device)
    
    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]
    return tokenizers, text_encoders

def run_encode_prompt():
    data_dir = [FLICKR2K_PATH, DIV2K_PATH, LSDIR20K_PATH, FFHQ10K_PATH ]
    hr_data_file, lr_data_file = merge_data(data_dir)
    for data_file in data_dir:
        if not os.path.exists(os.path.join(data_file, "prompt_embeds")):
            os.makedirs(os.path.join(data_file, "prompt_embeds"))
        if not os.path.exists(os.path.join(data_file, "pool_embeds")):
            os.makedirs(os.path.join(data_file, "pool_embeds"))
            
    tokenizers ,text_encoders = import_text_encoder(device="cuda")

    for hr_img_file in tqdm(hr_data_file, total=len(hr_data_file)):
        prompt_file = hr_img_file.replace(".png",".txt").replace("gt", "gt_DAPE")
        prompt_path = prompt_file.replace(".txt",".pt").replace("gt_DAPE", "prompt_embeds")
        pool_path = prompt_file.replace(".txt",".pt").replace("gt_DAPE", "pool_embeds")
        
        with open(prompt_file, "r") as f:
            prompt = f.read()
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, device="cuda")
        torch.save(prompt_embeds.detach().cpu(), prompt_path)    
        torch.save(pooled_prompt_embeds.detach().cpu(), pool_path)    
        print("{} Done !".format(prompt_path))
        
        
def run_encode_val_prompt():
    data_dir = [FLICKR2K_PATH, DIV2K_PATH, LSDIR20K_PATH, FFHQ10K_PATH ]
    lr_data_file = merge_val_data(data_dir)

    for data_file in data_dir:
        if not os.path.exists(os.path.join(data_file, "prompt_embeds")):
            os.makedirs(os.path.join(data_file, "prompt_embeds"))
        if not os.path.exists(os.path.join(data_file, "pool_embeds")):
            os.makedirs(os.path.join(data_file, "pool_embeds"))
            
    tokenizers ,text_encoders = import_text_encoder(device="cuda")

    for hr_img_file in tqdm(lr_data_file, total=len(lr_data_file)):
        prompt_file = hr_img_file.replace(".png",".txt").replace("test_LR", "DAPE")
        prompt_path = prompt_file.replace(".txt",".pt").replace("DAPE", "prompt_embeds")
        pool_path = prompt_file.replace(".txt",".pt").replace("DAPE", "pool_embeds")
        
        with open(prompt_file, "r") as f:
            prompt = f.read()
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, device="cuda")
        torch.save(prompt_embeds.detach().cpu(), prompt_path)    
        torch.save(pooled_prompt_embeds.detach().cpu(), pool_path)    
        print("{} Done !".format(prompt_path))
        
# ---------------------- vae encode ---------------------- #
def vae_encode(lr_img_paths, hr_img_paths,lr_latent_path="latent_lr",hr_latent_path="latent_hr" ,sd3_model_path="defualt", weight_dtype=torch.float32):
    vae = AutoencoderKL.from_pretrained(sd3_model_path, subfolder="vae").to("cuda", weight_dtype)
    with torch.no_grad():
        for lr_img_file,hr_img_file in tqdm(zip(lr_img_paths, hr_img_paths), total=len(lr_img_paths)):
            lr_save_path = lr_img_file.replace(".png",".pt").replace("sr_bicubic", lr_latent_path)
            hr_save_path = hr_img_file.replace(".png",".pt").replace("gt", hr_latent_path)
            lr_img = Image.open(lr_img_file).convert("RGB")
            hr_img = Image.open(hr_img_file).convert("RGB")
            trans = transforms.ToTensor()
            lq = trans(lr_img).unsqueeze(0).to("cuda",dtype=weight_dtype) * 2 - 1
            hq = trans(hr_img).unsqueeze(0).to("cuda",dtype=weight_dtype) * 2 - 1
            lq_latent = vae.encode(lq).latent_dist.sample() * vae.config.scaling_factor
            hq_latent = vae.encode(hq).latent_dist.sample() * vae.config.scaling_factor
            torch.save(lq_latent.detach().cpu(), lr_save_path)
            torch.save(hq_latent.detach().cpu(), hr_save_path)
            del lq_latent, hq_latent ,lq, hq
            
            print("{} Done !".format(lr_save_path.split("/")[-1]))

def vae_encode_down(lr_img_paths, hr_img_paths, sd3_model_path="checkpoint/tinybackbone/prune-12-merge-tinysr", weight_dtype=torch.float32, target_dir="256_path", scale=0.5):
    vae = AutoencoderKL.from_pretrained(sd3_model_path, subfolder="vae").to("cuda", weight_dtype)
    with torch.no_grad():
        for lr_img_file,hr_img_file in tqdm(zip(lr_img_paths, hr_img_paths), total=len(lr_img_paths)):
            save_path = lr_img_file.replace(".png",".pt").replace("sr_bicubic", target_dir)
            print(save_path)
            lr_img = Image.open(lr_img_file).convert("RGB")
            hr_img = Image.open(hr_img_file).convert("RGB")

            trans = transforms.ToTensor()
            lq = trans(lr_img).unsqueeze(0).to("cuda",dtype=weight_dtype) * 2 - 1
            hq = trans(hr_img).unsqueeze(0).to("cuda",dtype=weight_dtype) * 2 - 1
            
            lq = torch.nn.functional.interpolate(lq, scale_factor=scale, mode="bicubic", align_corners=False)
            hq = torch.nn.functional.interpolate(hq, scale_factor=scale, mode="bicubic", align_corners=False)
            
            lq_latent = vae.encode(lq).latent_dist.sample() * vae.config.scaling_factor
            hq_latent = vae.encode(hq).latent_dist.sample() * vae.config.scaling_factor
            
            latent = lq_latent - hq_latent
            torch.save(latent.detach().cpu(), save_path)
            del lq_latent, hq_latent ,lq, hq, latent
            
            print("{} Done !".format(save_path.split("/")[-1]))

def run_vae_encode():
    data_dir = [FLICKR2K_PATH, DIV2K_PATH, LSDIR20K_PATH, FFHQ10K_PATH ]
    # data_dir = [FLICKR2K_PATH ]
    
    hr_data_file, lr_data_file = merge_data(data_dir)
    for data_file in data_dir:
        if not os.path.exists(os.path.join(data_file, "latent_lr")):
            os.makedirs(os.path.join(data_file, "latent_lr"))
        if not os.path.exists(os.path.join(data_file, "latent_hr")):
            os.makedirs(os.path.join(data_file, "latent_hr"))
    
    vae_encode(lr_data_file, hr_data_file)

def run_vae_encode_down(variant="256"):
    data_dir = [FLICKR2K_PATH, DIV2K_PATH, FFHQ10K_PATH ]
    
    hr_data_file, lr_data_file = merge_data(data_dir)
    dir_name = f"{variant}_path"
    scale = 0.25 if variant == "128" else 0.5
    for data_file in data_dir:
        if not os.path.exists(os.path.join(data_file, dir_name)):
            os.makedirs(os.path.join(data_file, dir_name))


    vae_encode_down(lr_data_file, hr_data_file, target_dir=dir_name, scale=scale)

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else "256"
    run_vae_encode_down(variant=target)
    # run_encode_prompt()
    # run_encode_val_prompt()
    print("Done !")

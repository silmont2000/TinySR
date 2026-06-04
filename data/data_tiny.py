import os
import sys
sys.path.append(os.getcwd())
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


DIV2K_PATH = "/data/disk3/xby/tinysr/datasets/CAD_V100_20260209_backup/datasets/DIV2K/DIV2K_train_LRx4_Real-ESRGAN_Seesr_v2"
FLICKR2K_PATH =  "/data/disk3/xby/tinysr/datasets/CAD_V100_20260209_backup/datasets/Flickr2K/Flickr2K_LRx4_Real-ESRGAN_Seesr_v2"
FFHQ10K_PATH =  "/data/disk3/xby/tinysr/datasets/CAD_V100_20260209_backup/datasets/FFHQ/FFHQ10K_LRx4_Real-ESRGAN_Seesr_v2"


data_path = [DIV2K_PATH, FLICKR2K_PATH, FFHQ10K_PATH]
lr_dir_name = "sr_bicubic"
hr_dir_name = "gt"
prompt_dir_name = "gt_DAPE"
prompt_embeds_dir_name = "prompt_embeds"
pool_prompt_embeds_dir_name = "pool_embeds"
hr_latnet_dir_name = "latent_hr"

class SmokeDataset(Dataset):
    """5-image smoke test dataset, same format as smoke_overfit.py."""
    def __init__(self, base="dataset/smoke"):
        self.files = sorted(os.listdir(os.path.join(base, "latent_stu")))
        self.vae_dir = os.path.join(base, "vae_stu")
        self.stu_dir = os.path.join(base, "latent_stu")
        self.pool_dir = os.path.join(base, "pool_embeds")
        self.teacher_32_dir = os.path.join(base, "latent_stu_teacher_32")
        self.teacher_32_qkv_dir = os.path.join(base, "latent_stu_teacher_32_qkv_12")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        result = {
            "vae_stu": torch.load(os.path.join(self.vae_dir, f), map_location="cpu").squeeze(0),
            "latent_stu": torch.load(os.path.join(self.stu_dir, f), map_location="cpu").squeeze(0),
            "pooled_prompt_embeds_input": torch.load(os.path.join(self.pool_dir, f), map_location="cpu").squeeze(0),
        }
        teacher_32_path = os.path.join(self.teacher_32_dir, f)
        if os.path.exists(teacher_32_path):
            result["latent_stu_teacher_32"] = torch.load(teacher_32_path, map_location="cpu")
        qkv_path = os.path.join(self.teacher_32_qkv_dir, f)
        if os.path.exists(qkv_path):
            result["latent_stu_teacher_32_qkv"] = torch.load(qkv_path, map_location="cpu")
        return result


class Real_ESRGAN_Dataset(Dataset):
    def __init__(self,
                root_dir_path=data_path, 
                process_size=512,
                max_sample=None,
                device="cpu",
                first_stage=False,
                ):
        self.root_path=data_path
        self.first_stage = first_stage
        self.device = device
        self.max_sample = max_sample
        self.process_size=process_size
        self.lr_dir_name = lr_dir_name
        self.hr_dir_name = hr_dir_name
        self.hr_latnet_dir_name = hr_latnet_dir_name
        self.prompt_dir_name = prompt_dir_name
        self.prompt_embeds_dir_name = prompt_embeds_dir_name
        self.pool_prompt_embeds_dir_name = pool_prompt_embeds_dir_name
        self.trans = transforms.ToTensor()
        
        self.lr_img_name = []
        self.hr_img_name = []
        self.lr_latent_name = []
        self.hr_latent_name = []
        self.prompt_name = []
        self.prompt_embeds_name = []
        self.pool_prompt_embeds_name = []
        
        for root_dir in root_dir_path:
            # image data path
            lr_data_path = os.path.join(root_dir, self.lr_dir_name)
            hr_data_path = os.path.join(root_dir, self.hr_dir_name)
            latent_hr_path = os.path.join(root_dir, self.hr_latnet_dir_name)
            # prompt path
            self.prompt_path = os.path.join(root_dir, self.prompt_dir_name)
            self.prompt_embeds_path = os.path.join(root_dir, self.prompt_embeds_dir_name)
            self.pool_prompt_embeds_path = os.path.join(root_dir, self.pool_prompt_embeds_dir_name)
            
            data_file = os.listdir(lr_data_path)
            data_file.sort(key=lambda x: int(x.split(".")[0]))
            
            self.lr_img_name = self.lr_img_name + [os.path.join(lr_data_path, file) for file in data_file]
            self.hr_img_name = self.hr_img_name + [os.path.join(hr_data_path, file) for file in data_file]
            self.hr_latent_name = self.hr_latent_name + [os.path.join(latent_hr_path, file.replace(".png", ".pt")) for file in data_file]
            
            self.prompt_name = self.prompt_name + [os.path.join(self.prompt_path, file.replace(".png", ".txt")) for file in data_file]
            self.prompt_embeds_name = self.prompt_embeds_name + [os.path.join(self.prompt_embeds_path, file.replace(".png", ".pt")) for file in data_file]
            self.pool_prompt_embeds_name = self.pool_prompt_embeds_name + [os.path.join(self.pool_prompt_embeds_path, file.replace(".png", ".pt")) for file in data_file]
            
        self.img_nums = len(self.lr_img_name)
    
    def __getitem__(self, idx):

        latent_stu = torch.load(self.hr_latent_name[idx].replace("latent_hr", "latent_stu"), map_location=self.device).squeeze() 
        # vae_stu = torch.load(self.hr_latent_name[idx].replace("latent_hr", "vae_stu_lr_256"), map_location=self.device).squeeze()
        vae_stu = torch.load(self.hr_latent_name[idx].replace("latent_hr", "vae_stu"), map_location=self.device).squeeze()
        pooled_prompt_embeds = torch.load(self.pool_prompt_embeds_name[idx], map_location=self.device).squeeze()

        if self.first_stage == True:
            result = {
                "latent_stu": latent_stu,
                "vae_stu": vae_stu,
                "pooled_prompt_embeds_input": pooled_prompt_embeds,
            }

        else:
            latent_hr = torch.load(self.hr_latent_name[idx], map_location=self.device).squeeze() 
            img_names = self.lr_img_name[idx]
            lr_img = self.trans(Image.open(self.lr_img_name[idx]).convert("RGB")).squeeze() * 2 - 1 
            hr_img = self.trans(Image.open(self.hr_img_name[idx]).convert("RGB")).squeeze() * 2 - 1 
        
            result = {
                "img_name": img_names,
                "lr_img": lr_img,
                "hr_img": hr_img,
                "latent_hr": latent_hr,
                "latent_stu": latent_stu,
                "vae_stu": vae_stu,
                "pooled_prompt_embeds_input": pooled_prompt_embeds,
            }

        teacher_32_path = self.hr_latent_name[idx].replace("latent_hr", "latent_stu_teacher_32")
        if os.path.exists(teacher_32_path):
            result["latent_stu_teacher_32"] = torch.load(teacher_32_path, map_location=self.device)
        qkv_path = self.hr_latent_name[idx].replace("latent_hr", "latent_stu_teacher_32_qkv_12")
        if os.path.exists(qkv_path):
            result["latent_stu_teacher_32_qkv"] = torch.load(qkv_path, map_location=self.device)

        return result

    
    def __len__(self):
        if self.max_sample:
            return self.max_sample
        return self.img_nums
   
if __name__ == "__main__":
    dataset = Real_ESRGAN_Dataset()
    for i in range(len(dataset)):
        print(len(dataset))
        data = dataset[i]
        print(data["lr_img"].shape)  # [3, 512, 512]
        print(data["hr_img"].shape)  # [3, 512, 512]
        print(data["latent_stu"].shape) # [16, 64, 64]
        print(data["vae_stu"].shape) # [333, 4096]
        print(data["pooled_prompt_embeds_input"].shape) # [2048]
        break
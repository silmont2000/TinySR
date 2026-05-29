import sys, os, argparse, json
sys.path.append(".")

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from peft.utils import get_peft_model_state_dict
from diffusers import StableDiffusion3Pipeline
from models.tinysr.stage1_defaults import (
    CKPT, DEFAULT_PYRAMID_CONFIG, LORA_R, SMOKE_RANK_PATTERN, make_lora_config,
    DEFAULT_TIMESTEP,
)
from models.tinysr.pyramid_model import TinyPyramidSD3Transformer2DModel
from datetime import datetime

STEPS = 2000
LR = 1e-4

# Usage:
#   CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 smoke_overfit.py


class SmokeDataset(Dataset):
    def __init__(self, scheme):
        base = "dataset/smoke"
        self.files = sorted(os.listdir(os.path.join(base, "latent_stu")))
        self.vae_dir = os.path.join(base, "vae_stu_lr")
        self.stu_dir = os.path.join(base, "latent_stu")
        self.pool_dir = os.path.join(base, "pool_embeds")
        if scheme == "1":
            self.tgt4_dir = os.path.join(base, "latent_stu_4x")
            self.tgt2_dir = os.path.join(base, "latent_stu_2x")
        else:
            base2 = "dataset/smoke_scheme2"
            self.tgt4_dir = os.path.join(base2, "target_8")
            self.tgt2_dir = os.path.join(base2, "target_16")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        vae = torch.load(os.path.join(self.vae_dir, f), map_location="cpu").squeeze(0)
        stu = torch.load(os.path.join(self.stu_dir, f), map_location="cpu").squeeze(0)
        stu4 = torch.load(os.path.join(self.tgt4_dir, f), map_location="cpu")
        stu2 = torch.load(os.path.join(self.tgt2_dir, f), map_location="cpu")
        pool = torch.load(os.path.join(self.pool_dir, f), map_location="cpu").squeeze(0)
        if stu4.dim() == 4:
            stu4 = stu4.squeeze(0)
        if stu2.dim() == 4:
            stu2 = stu2.squeeze(0)
        return {"vae_stu": vae, "latent_stu": stu, "tg4": stu4, "tg2": stu2, "pool": pool}


def save_ckpt(accelerator, model, args, step=None):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{step}" if step is not None else ""
        ckpt_dir = f"outputs/smoke_checkpoint_s{args.scheme}_{current_time}{suffix}"
        os.makedirs(ckpt_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        with open(os.path.join(ckpt_dir, "pyramid_config.json"), "w") as f:
            json.dump(unwrapped.pyramid_config.to_dict(), f, indent=2)
        with open(os.path.join(ckpt_dir, "rank_pattern.json"), "w") as f:
            json.dump(SMOKE_RANK_PATTERN, f, indent=2)
        with open(os.path.join(ckpt_dir, "train_config.json"), "w") as f:
            json.dump({"loss_type": args.loss_type}, f, indent=2)
        # for n, p in unwrapped.named_parameters():
        #     if "norm1" in n:
        #         print(n, p.requires_grad, p.shape)
        lora_state = get_peft_model_state_dict(unwrapped, adapter_name="default")

        # print("LoRA keys count:", len(lora_state))
        # for k in lora_state:
        #     if 'norm1' in k:
        #         print("Found:", k)
                
        StableDiffusion3Pipeline.save_lora_weights(
            ckpt_dir, transformer_lora_layers=lora_state,
            weight_name="transformer.safetensors")
        print(f"Saved: {ckpt_dir}")


def main(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="fp16",kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    dtype = torch.float16
    pc = DEFAULT_PYRAMID_CONFIG

    # ── Model ──
    model = TinyPyramidSD3Transformer2DModel.from_flat_pretrained(
        CKPT, pyramid_config=pc, subfolder="transformer", torch_dtype=dtype,
    )
    model.requires_grad_(False)

    # ── LoRA ──
    if accelerator.is_main_process:
        import wandb
        wandb.init(project="tinysr", name=f"smoke-s{args.scheme}-lt{args.loss_type}",
                   config={
                       "pyramid_config": pc.to_dict(),
                       "lora_r": LORA_R,
                       "rank_pattern": SMOKE_RANK_PATTERN,
                       "scheme": args.scheme,
                       "loss_type": args.loss_type,
                       "num_processes": accelerator.num_processes,
                   }, mode="online")
        print(f"Scheme {args.scheme}, loss_type={args.loss_type}, {accelerator.num_processes} GPUs")
        print(f"  sample_size={pc.sample_size}, need_down_proj={pc.need_down_proj}, upsample_mode={pc.upsample_mode}")

    lora_cfg = make_lora_config(rank_pattern=SMOKE_RANK_PATTERN)
    model.add_adapter(lora_cfg, adapter_name="default")
    model.enable_adapters()

    # ── Training ──
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR)
    ds = SmokeDataset(args.scheme)
    loader = DataLoader(ds, batch_size=8, shuffle=True)

    model, opt, loader = accelerator.prepare(model, opt, loader)
    model.train()

    if accelerator.is_main_process:
        print(f"Dataset: {len(ds)} samples, {STEPS} steps")

    t_step = torch.tensor([DEFAULT_TIMESTEP], device=device).long()
    data_iter = iter(loader)

    for step in range(1, STEPS + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        opt.zero_grad()
        mi, tg, pl = batch["vae_stu"], batch["latent_stu"], batch["pool"]

        with accelerator.autocast():
            out, pre_last = model(hidden_states=mi, timestep=t_step, pooled_projections=pl, return_dict=False)

            if args.loss_type == "a":
                scale_factor = pc.p_states[-1].grid_hw // pc.p_states[0].grid_hw
                input_up = F.interpolate(mi.float(), scale_factor=scale_factor, mode='bilinear', align_corners=False)
                denoised = input_up - out.float()
                loss = F.l1_loss(denoised, tg.float().detach())
            else:
                last_grid_hw = pc.p_states[-1].grid_hw
                latent_before = model._tokens_to_latent(pre_last.float(), last_grid_hw)
                denoised = latent_before.float() - out.float()
                loss = F.l1_loss(denoised, tg.float().detach())

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(trainable, 0.5)
        opt.step()

        if step % 20 == 0 and accelerator.is_main_process:
            wandb.log({"l1_loss": loss.item()}, step=step)
        if step % 500 == 0 and accelerator.is_main_process:
            print(f"  step {step:5d} | L1={loss.item():.6f}")
        if step % 2000 == 2:
            save_ckpt(accelerator, model, args, step)
            accelerator.wait_for_everyone()

    # ── Save final checkpoint ──
    save_ckpt(accelerator, model, args)
    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", type=str, default="1", choices=["1", "2"])
    p.add_argument("--loss_type", type=str, default="a", choices=["a", "b"],
                   help="a: upsample input, subtract noise | b: proj_out(pre_last_tokens) - noise")
    main(p.parse_args())

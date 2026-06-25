import sys, os, argparse
sys.path.append(".")
import torch
from PIL import Image
from torchvision import transforms

from models.vae.autoencoder_tiny import AutoencoderTiny
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from utils.device import get_optimal_device_name

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lora_dir", type=str, required=True,
                   help="flat stage1 checkpoint dir")
    p.add_argument("--input_image", type=str, default="dataset/test_image/Canon_10_x1.png")
    p.add_argument("--output_dir", type=str, default="outputs/validate_flat")
    p.add_argument("--device", type=str, default=get_optimal_device_name())
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_lora", action="store_true", help="skip LoRA, run base model only")
    return p.parse_args()

def main():
    args = parse_args()
    device = args.device
    dtype = torch.float16
    torch.manual_seed(args.seed)

    print("Loading VAE...")
    vae = AutoencoderTiny.from_pretrained("checkpoint/vae/separable")
    vae.to(device, dtype=dtype)
    vae.requires_grad_(False)

    print("Loading flat TinySD3 model...")
    model = TinySD3Transformer2DModel.from_pretrained(
        "checkpoint/tinybackbone/prune-12-merge-tinysr",
        subfolder="transformer",
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        torch_dtype=dtype,
    )
    model.requires_grad_(False)

    if not args.no_lora and os.path.isdir(args.lora_dir):
        from peft import LoraConfig
        from peft.utils import set_peft_model_state_dict

        lora_cfg = LoraConfig(
            r=64, lora_alpha=64, init_lora_weights="gaussian",
            target_modules=["to_k","to_q","to_v","to_out.0","proj","linear","linear_1","linear_2","net.2"],
        )
        model.add_adapter(lora_cfg, adapter_name="default")
        model.enable_adapters()

        # lora_file = os.path.join(args.lora_dir, "transformer.safetensors")
        lora_file = os.path.join(args.lora_dir, "model.safetensors")
        import safetensors.torch
        raw_sd = safetensors.torch.load_file(lora_file)
        lora_sd = {k.replace("transformer.", ""): v for k, v in raw_sd.items()}
        missing, unexpected = set_peft_model_state_dict(model, lora_sd, adapter_name="default")
        print(f"LoRA loaded: {len(missing)} missing, {len(unexpected)} unexpected")

        # debug: print first LoRA A weight stats
        for k, v in lora_sd.items():
            if "lora_A" in k and "to_q" in k:
                print(f"  [{k}] mean={v.mean():.6f} std={v.std():.6f}")
                break

    model.to(device, dtype=dtype)
    model.eval()

    print(f"Loading image: {args.input_image}")
    img = Image.open(args.input_image).convert("RGB")
    img_resized = img.resize((512, 512), Image.BICUBIC)

    img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(device, dtype=dtype)
    img_tensor = img_tensor * 2 - 1

    with torch.no_grad():
        latent = vae.encode(img_tensor).latents * vae.config.scaling_factor

    timestep = torch.tensor([1000], device=device).long()
    pooled = torch.load("dataset/default/pool_embeds.pt", map_location=device).to(dtype)

    with torch.no_grad():
        pred = model(
            hidden_states=latent, timestep=timestep,
            pooled_projections=pooled, return_dict=False,
        )[0]
        refined = latent - pred
        output = vae.decode(refined / vae.config.scaling_factor, return_dict=False)[0]
        output = output.clamp(-1, 1).squeeze(0)

    output = (output * 0.5 + 0.5).clamp(0, 1)
    output_img = transforms.ToPILImage()(output.float().cpu())

    os.makedirs(args.output_dir, exist_ok=True)
    tag = os.path.basename(args.lora_dir.rstrip("/"))
    out_path = os.path.join(args.output_dir, f"flat_{tag}.png")

    side_by_side = Image.new("RGB", (1024, 512))
    side_by_side.paste(img_resized, (0, 0))
    side_by_side.paste(output_img, (512, 0))
    side_by_side.save(out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()

import sys, os, argparse, json
sys.path.append(".")
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.vae.autoencoder_tiny import AutoencoderTiny
from models.tinysr.pyramid_config import PyramidArchConfig, PStateSpec
from models.tinysr.pyramid_model import TinyPyramidSD3Transformer2DModel
# cd /data/disk2/xby/TinySR
# conda activate tinysr

# # 看最新的 checkpoint
# python validate_stage1.py \
#   --lora_dir checkpoint/pyramid-stage1/checkpoint-25500 \
#   --output_dir outputs/validate_stage1

# # 换别的图
# python validate_stage1.py \
#   --lora_dir checkpoint/pyramid-stage1/checkpoint-20500 \
#   --input_image dataset/test_image/0801_pch_00001.png \
#   --output_dir outputs/validate_stage1

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lora_dir", type=str, default="checkpoint/pyramid-stage1/checkpoint-5000",
                   help="path to stage1 lora checkpoint dir")
    p.add_argument("--input_image", type=str, default="dataset/test_image/Canon_10_x1.png")
    p.add_argument("--output_dir", type=str, default="outputs/validate_stage1")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    dtype = torch.float16
    torch.manual_seed(args.seed)

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderTiny.from_pretrained("checkpoint/vae/separable")
    vae.to(device, dtype=dtype)
    vae.requires_grad_(False)

    # Load pyramid model
    print("Loading pyramid model...")
    cfg_path = os.path.join(args.lora_dir, "pyramid_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg_dict = json.load(f)
        pc = PyramidArchConfig.from_dict(cfg_dict)
        print(f"  Loaded config from {cfg_path}: sample_size={pc.sample_size}, {[(s.num_blocks,s.dim,s.grid_hw) for s in pc.p_states]}")
    else:
        pc = PyramidArchConfig(
            p_states=(
                PStateSpec(num_blocks=4, dim=1536, grid_hw=8),
                PStateSpec(num_blocks=4, dim=1536, grid_hw=16),
                PStateSpec(num_blocks=4, dim=1536, grid_hw=32),
            )
        )
        print(f"  Using default config (sample_size={pc.sample_size})")

    model = TinyPyramidSD3Transformer2DModel.from_flat_pretrained(
        "checkpoint/tinybackbone/prune-12-merge-tinysr",
        pyramid_config=pc,
        subfolder="transformer",
        torch_dtype=dtype,
    )

    # Load Stage 1 LoRA
    if os.path.isdir(args.lora_dir) and "checkpoint" in args.lora_dir:
        from peft import LoraConfig
        from peft.utils import set_peft_model_state_dict

        lora_cfg = LoraConfig(
            r=64, lora_alpha=64, init_lora_weights="gaussian",
            target_modules=["to_k","to_q","to_v","to_out.0","proj","linear","linear_1","linear_2","net.2"],
            rank_pattern={
                "p_states.0": 16,
                "p_states.1": 64,
                "p_states.2": 256,
            },
        )
        model.add_adapter(lora_cfg, adapter_name="default")
        model.enable_adapters()

        lora_file = os.path.join(args.lora_dir, "transformer.safetensors")
        if os.path.exists(lora_file):
            import safetensors.torch
            raw_sd = safetensors.torch.load_file(lora_file)
            # Strip "transformer." prefix from keys
            lora_sd = {k.replace("transformer.", ""): v for k, v in raw_sd.items()}
            missing, unexpected = set_peft_model_state_dict(model, lora_sd, adapter_name="default")
            print(f"LoRA loaded from {os.path.basename(args.lora_dir)}: {len(missing)} missing, {len(unexpected)} unexpected")
        else:
            print(f"Warning: no transformer.safetensors in {args.lora_dir}")
    else:
        print(f"Warning: {args.lora_dir} not found, using base model")

    model.to(device, dtype=dtype)
    model.eval()

    # Load image
    print(f"Loading image: {args.input_image}")
    img = Image.open(args.input_image).convert("RGB")
    w, h = img.size

    process_size = pc.sample_size * 8  # VAE reduces 8×, so image size = sample_size * 8
    img_resized = img.resize((process_size, process_size), Image.BICUBIC)

    img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(device, dtype=dtype)
    img_tensor = img_tensor * 2 - 1  # normalize to [-1, 1]

    # VAE encode
    with torch.no_grad():
        latent = vae.encode(img_tensor).latents * vae.config.scaling_factor

    # Model forward
    timestep = torch.tensor([1000], device=device).long()
    pooled = torch.load("dataset/default/pool_embeds.pt", map_location=device).to(dtype)

    with torch.no_grad():
        output, stage_noises, stage_inputs = model(
            hidden_states=latent,
            timestep=timestep,
            pooled_projections=pooled,
            return_dict=False,
        )
        refined = stage_inputs[2] - output

        # VAE decode
        output_img = vae.decode(refined / vae.config.scaling_factor, return_dict=False)[0]
        output_img = output_img.clamp(-1, 1).squeeze(0)

    # Convert to PIL
    output_img = (output_img * 0.5 + 0.5).clamp(0, 1)  # [-1,1] → [0,1]
    output_pil = transforms.ToPILImage()(output_img.float().cpu())

    # Save side by side
    os.makedirs(args.output_dir, exist_ok=True)
    tag = os.path.basename(args.lora_dir.rstrip("/"))
    out_name = f"validate_{tag}.png"
    out_path = os.path.join(args.output_dir, out_name)

    # Side by side: input(left) | output(right)
    side_by_side = Image.new("RGB", (1024, 512))
    side_by_side.paste(img_resized, (0, 0))
    side_by_side.paste(output_pil, (512, 0))
    side_by_side.save(out_path)
    print(f"Saved: {out_path}")
    print(f"  Left: input  |  Right: model output")


if __name__ == "__main__":
    main()

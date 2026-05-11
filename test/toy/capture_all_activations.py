import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusion3Pipeline

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Capture all layer activations for TinySR on a real LQ image.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="checkpoint/tinybackbone/prune-12-merge-tinysr")
    parser.add_argument("--vae_path", type=str, default="checkpoint/vae/separable")
    parser.add_argument("--lora_dir", type=str, default="")
    parser.add_argument("--embedding_dir", type=str, default="dataset/default/")
    parser.add_argument("--input_image", type=str, default="dataset/test_image/Canon_10_x1.png")
    parser.add_argument("--out_dir", type=str, default="outputs/activations_capture")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument(
        "--capture_mode",
        type=str,
        choices=["forward", "forward_pre"],
        default="forward",
        help="forward captures module outputs; forward_pre captures module inputs.",
    )
    parser.add_argument("--save_output_image", action="store_true")
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_.-]", "_", name)


def is_leaf_module(module: torch.nn.Module) -> bool:
    return len(list(module.children())) == 0


def to_serializable_activation(output, cast_fp16=True):
    def _convert_tensor(t):
        t = t.detach().cpu()
        if cast_fp16 and torch.is_floating_point(t):
            t = t.to(torch.float16)
        return t

    if isinstance(output, torch.Tensor):
        return _convert_tensor(output)
    if isinstance(output, (tuple, list)):
        tensors = [_convert_tensor(x) for x in output if isinstance(x, torch.Tensor)]
        return tensors if tensors else None
    if isinstance(output, dict):
        tensors = {k: _convert_tensor(v) for k, v in output.items() if isinstance(v, torch.Tensor)}
        return tensors if tensors else None
    return None


def register_hooks(model, model_tag, save_dir, manifest, cast_fp16=True, capture_mode="forward"):
    hooks = []
    call_counter = {"idx": 0}

    for module_name, module in model.named_modules():
        if not is_leaf_module(module):
            continue

        full_name = f"{model_tag}.{module_name}" if module_name else model_tag

        def save_activation(raw_value, name):
            act = to_serializable_activation(raw_value, cast_fp16=cast_fp16)
            if act is None:
                return
            call_idx = call_counter["idx"]
            call_counter["idx"] += 1

            file_name = f"{call_idx:05d}_{sanitize_name(name)}.pt"
            file_path = save_dir / file_name
            torch.save(act, file_path)

            if isinstance(act, torch.Tensor):
                shape_info = [list(act.shape)]
                dtype_info = [str(act.dtype)]
            elif isinstance(act, list):
                shape_info = [list(x.shape) for x in act]
                dtype_info = [str(x.dtype) for x in act]
            else:
                shape_info = {k: list(v.shape) for k, v in act.items()}
                dtype_info = {k: str(v.dtype) for k, v in act.items()}

            manifest.append(
                {
                    "call_idx": call_idx,
                    "layer": name,
                    "file": str(file_path),
                    "shape": shape_info,
                    "dtype": dtype_info,
                    "capture_mode": capture_mode,
                }
            )

        def hook_forward(_module, _inputs, output, name=full_name):
            save_activation(output, name)

        def hook_forward_pre(_module, inputs, name=full_name):
            if isinstance(inputs, (tuple, list)) and len(inputs) > 0:
                save_activation(inputs[0], name)
            else:
                save_activation(inputs, name)

        if capture_mode == "forward_pre":
            hooks.append(module.register_forward_pre_hook(hook_forward_pre))
        else:
            hooks.append(module.register_forward_hook(hook_forward))
    return hooks


def load_models(args, weight_dtype, device):
    transformer = TinySD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
    )
    vae = AutoencoderTiny.from_pretrained(args.vae_path, torch_dtype=weight_dtype)

    if args.lora_dir:
        from peft import LoraConfig

        transformer_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj", "linear", "linear_1", "linear_2", "net.2"],
        )
        transformer.add_adapter(transformer_lora_config)
        transformer.enable_adapters()
        transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(
            args.lora_dir, weight_name="transformer.safetensors"
        )
        load_lora_state_dict(transformer_lora_state_dict, transformer)

    vae = vae.to(device, dtype=weight_dtype).eval()
    transformer = transformer.to(device, dtype=weight_dtype).eval()
    return transformer, vae


def preprocess_image(args, device, weight_dtype):
    image = Image.open(args.input_image).convert("RGB")
    ori_width, ori_height = image.size

    resize_flag = False
    if ori_width < args.process_size // args.upscale or ori_height < args.process_size // args.upscale:
        scale = (args.process_size // args.upscale) / min(ori_width, ori_height)
        new_width, new_height = int(scale * ori_width), int(scale * ori_height)
        resize_flag = True
    else:
        new_width, new_height = ori_width, ori_height

    new_width, new_height = args.upscale * new_width, args.upscale * new_height
    if new_width % 8 or new_height % 8:
        resize_flag = True
        new_width -= new_width % 8
        new_height -= new_height % 8

    pixel_values = transforms.ToTensor()(image).unsqueeze(0)
    pixel_values = torch.nn.functional.interpolate(
        pixel_values, size=(new_height, new_width), mode="bicubic", align_corners=False
    )
    pixel_values = (pixel_values * 2 - 1).to(device, dtype=weight_dtype)

    return image, pixel_values, resize_flag


def main():
    args = parse_args()
    os.chdir(Path(__file__).resolve().parents[1])

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，回退到 CPU。")
        args.device = "cpu"

    device = torch.device(args.device)
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

    # 某些缓存张量是在 CUDA 设备上保存的，CPU 环境下强制映射到 CPU。
    if device.type == "cpu":
        _orig_torch_load = torch.load

        def _patched_torch_load(*load_args, **load_kwargs):
            load_kwargs.setdefault("map_location", "cpu")
            return _orig_torch_load(*load_args, **load_kwargs)

        torch.load = _patched_torch_load

    out_dir = Path(args.out_dir).resolve()
    act_dir = out_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    transformer, vae = load_models(args, weight_dtype, device)
    pooled_prompt_embeds = torch.load(
        os.path.join(args.embedding_dir, "pool_embeds.pt"), map_location=device
    ).to(dtype=weight_dtype)
    timesteps = torch.tensor([1000.0], device=device, dtype=weight_dtype)

    src_image, pixel_values, _ = preprocess_image(args, device, weight_dtype)

    manifest = []
    hooks = []
    hooks.extend(register_hooks(vae, "vae", act_dir, manifest, cast_fp16=True, capture_mode=args.capture_mode))
    hooks.extend(
        register_hooks(transformer, "transformer", act_dir, manifest, cast_fp16=True, capture_mode=args.capture_mode)
    )

    start_t = time.time()
    with torch.no_grad():
        model_input = vae.encode(pixel_values).latents * vae.config.scaling_factor
        model_pred = transformer(
            hidden_states=model_input,
            timestep=timesteps,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
        latent_stu = model_input - model_pred
        image = vae.decode(latent_stu / vae.config.scaling_factor, return_dict=False)[0].squeeze(0).clamp(-1, 1)
    elapsed = time.time() - start_t

    for h in hooks:
        h.remove()

    if args.save_output_image:
        out_img = transforms.ToPILImage()(image.cpu() / 2 + 0.5)
        out_img.save(out_dir / "reconstructed.png")

    run_meta = {
        "input_image": str(Path(args.input_image).resolve()),
        "device": str(device),
        "dtype": str(weight_dtype),
        "num_captured_activations": len(manifest),
        "elapsed_sec": elapsed,
        "args": vars(args),
        "source_image_size": list(src_image.size),
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"完成。捕获激活条目: {len(manifest)}")
    print(f"激活目录: {act_dir}")
    print(f"索引文件: {out_dir / 'manifest.json'}")
    print(f"运行信息: {out_dir / 'run_meta.json'}")


if __name__ == "__main__":
    main()

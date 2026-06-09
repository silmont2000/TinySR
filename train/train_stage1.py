#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import argparse
import datetime
import json
import logging
import math
import os
import sys
sys.path.append(os.getcwd())
import shutil
from contextlib import nullcontext
from pathlib import Path
import torch.nn.functional as F
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from data.data_tiny import Real_ESRGAN_Dataset
import diffusers
from diffusers import (
    StableDiffusion3Pipeline,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from models.tinysr.pyramid_config import PyramidArchConfig, PStateSpec
from models.tinysr.pyramid_model import TinyPyramidSD3Transformer2DModel
from models.tinysr.stage1_defaults import CKPT,SMOKE_RANK_PATTERN, make_lora_config, DEFAULT_PYRAMID_CONFIG, LORA_R, DEFAULT_TIMESTEP
if is_wandb_available():
    import wandb
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__)


def log_validation(
    log_dict,
    args,
    accelerator,
    vae_scale,
):
    logger.info(
        f"Running validation... \n Generating 3 images with prompt:"
    )

    images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        hq = torch.chunk(log_dict["hq"], args.train_batch_size, dim=0)[0]
        lq = torch.chunk(log_dict["lq"], args.train_batch_size, dim=0)[0]
        image_stu = torch.chunk(
            log_dict["image_stu"], args.train_batch_size, dim=0)[0]

        image_processor = VaeImageProcessor(
            vae_scale_factor=2 ** (vae_scale - 1))
        image = image_processor.postprocess(hq.detach())[0]
        images.append(image)
        image = image_processor.postprocess(lq.detach())[0]
        images.append(image)
        image = image_processor.postprocess(image_stu.detach())[0]
        images.append(image)

    caption = ["hq", "lq", "image_stu"]
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=caption[i])
                        for i, image in enumerate(images)
                    ]
                }
            )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--teacher_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to DINOv2 pretrained model.",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use 5-image smoke dataset for quick validation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd3-dreambooth",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--log_name",
        type=str,
        default="default-log",
        help="log_name",
    )
    parser.add_argument(
        "--wandb_id",
        type=str,
        default=None,
        help="wandb run to resume: run_id (e.g. 9at5m8zu), 'latest' for most recent run, or 'entity/project/run_id' path.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="wandb entity name, used when --wandb_id=latest.",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed for reproducible training.")
    parser.add_argument("--log_code", action="store_true", help="log code",)
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_d",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_discrimitor",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0,
                        help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--loss_type", type=str, default="a", choices=["a", "b"],
        help="a: upsample input, subtract noise | b: proj_out(pre_last_tokens) - noise",
    )
    parser.add_argument(
        "--cos_loss_weight", type=float, default=0.0,
        help="Weight for cosine similarity loss between student bridge output and teacher layer-5 output. 0 = disabled.",
    )
    parser.add_argument(
        "--attn_loss_weight", type=float, default=0.0,
        help="Weight for attention-distribution KL loss (L_AT). Requires precomputed latent_stu_teacher_32_attn/.",
    )
    parser.add_argument(
        "--vr_loss_weight", type=float, default=0.0,
        help="Weight for value-relation KL loss (L_VR). Requires precomputed latent_stu_teacher_32_vr/.",
    )
    parser.add_argument(
        "--attn_temperature", type=float, default=1.0,
        help="Temperature for softmax in attention/VR KL loss. >1 softens distribution, stabilizes gradients.",
    )
    parser.add_argument(
        "--cos_qkv_loss_weight", type=float, default=0.0,
        help="Weight for cosine-similarity loss on Q/K/V projections (alternative to attn/vr KL). More stable.",
    )
    parser.add_argument(
        "--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode"]
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=(
            'The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument("--adam_weight_decay", type=float,
                        default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0,
                        type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float16
    if accelerator.mixed_precision == "no":
        weight_dtype = torch.float32
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # ======================= 定义内部函数 =======================
    def resume_training():
        """返回 (global_step, pc, transformer_lora_config, pyramid_loss_type, checkpoint_path)"""
        if not args.resume_from_checkpoint:
            return 0, None, None, args.loss_type, None

        # ===== DEBUG: 断点续训路径解析日志 =====
        accelerator.print(f"[RESUME DEBUG] args.resume_from_checkpoint = '{args.resume_from_checkpoint}'")
        accelerator.print(f"[RESUME DEBUG] args.output_dir = '{args.output_dir}'")
        accelerator.print(f"[RESUME DEBUG]         abs = '{os.path.abspath(args.output_dir)}'")
        # ======================================

        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
            accelerator.print(f"[RESUME DEBUG] 非 latest 模式: basename('{args.resume_from_checkpoint}') = '{path}'")
        else:
            accelerator.print(f"[RESUME DEBUG] latest 模式: 在 output_dir 下查找 checkpoint-*")
            dirs = [d for d in os.listdir(
                args.output_dir) if d.startswith("checkpoint-")]
            if not dirs:
                accelerator.print(
                    "No checkpoint found. Starting a new training run.")
                accelerator.print(f"[RESUME DEBUG] output_dir 下无 checkpoint-* 目录")
                return 0, None, None, args.loss_type, None
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
            accelerator.print(f"[RESUME DEBUG] 找到的 checkpoint 目录: {dirs}")
            accelerator.print(f"[RESUME DEBUG] 选用最新: '{path}'")

        checkpoint_path = os.path.join(args.output_dir, path)
        accelerator.print(f"[RESUME DEBUG] 拼接结果 checkpoint_path = '{checkpoint_path}'")
        accelerator.print(f"[RESUME DEBUG]                         abs = '{os.path.abspath(checkpoint_path)}'")
        accelerator.print(f"[RESUME DEBUG] 路径是否存在: {os.path.exists(checkpoint_path)}")
        if not os.path.exists(checkpoint_path):
            accelerator.print(
                f"Checkpoint '{checkpoint_path}' does not exist. Starting a new training run.")
            accelerator.print(f"[RESUME DEBUG] !!! 将从头开始训练，不会续训 checkpoint-35601 !!!")
            return 0, None, None, args.loss_type, None

        accelerator.print(f"Resuming from checkpoint {path}")

        pc_path = os.path.join(checkpoint_path, "pyramid_config.json")
        if not os.path.exists(pc_path):
            raise FileNotFoundError(
                f"pyramid_config.json not found in {checkpoint_path}")
        with open(pc_path) as f:
            pc = PyramidArchConfig.from_dict(json.load(f))
        print(
            f"  pyramid: sample_size={pc.sample_size}, upsample={pc.upsample_mode}, {[(s.num_blocks,s.dim,s.grid_hw) for s in pc.p_states]}")

        rp_path = os.path.join(checkpoint_path, "rank_pattern.json")
        if os.path.exists(rp_path):
            with open(rp_path) as f:
                rank_pattern = json.load(f)
        else:
            rank_pattern = SMOKE_RANK_PATTERN
        transformer_lora_config = make_lora_config(rank_pattern=rank_pattern)

        tc_path = os.path.join(checkpoint_path, "train_config.json")
        if os.path.exists(tc_path):
            with open(tc_path) as f:
                pyramid_loss_type = json.load(f).get(
                    "loss_type", args.loss_type)
        else:
            pyramid_loss_type = args.loss_type
        print(f"  pyramid loss_type={pyramid_loss_type}")

        try:
            global_step = int(path.split("-")[1])
        except (IndexError, ValueError):
            global_step = 0

        return global_step, pc, transformer_lora_config, pyramid_loss_type, checkpoint_path

    def save_ckpt(step=None):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # 删除旧 checkpoint 以保持数量限制
            if args.checkpoints_total_limit is not None:
                checkpoints = [d for d in os.listdir(
                    args.output_dir) if d.startswith("checkpoint-")]
                checkpoints = sorted(
                    checkpoints, key=lambda x: int(x.split("-")[1]))
                if len(checkpoints) >= args.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - \
                        args.checkpoints_total_limit + 1
                    removing = checkpoints[:num_to_remove]
                    logger.info(
                        f"Removing {len(removing)} old checkpoints: {', '.join(removing)}")
                    for rem in removing:
                        shutil.rmtree(os.path.join(args.output_dir, rem))

            # 只使用 checkpoint-{global_step} 目录，把所有配置和状态存入其中
            save_path = os.path.join(
                args.output_dir, f"checkpoint-{global_step}")
            os.makedirs(save_path, exist_ok=True)

            # 保存模型配置文件
            unwrapped = accelerator.unwrap_model(transformer)
            with open(os.path.join(save_path, "pyramid_config.json"), "w") as f:
                json.dump(unwrapped.pyramid_config.to_dict(), f, indent=2)
            with open(os.path.join(save_path, "rank_pattern.json"), "w") as f:
                json.dump(SMOKE_RANK_PATTERN, f, indent=2)
            with open(os.path.join(save_path, "train_config.json"), "w") as f:
                json.dump({"loss_type": args.loss_type}, f, indent=2)

            # 保存完整训练状态（模型、优化器、调度器等）
            try:
                accelerator.save_state(save_path)
                logger.info(f"Saved checkpoint to {save_path}")
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM when saving checkpoint to {save_path}, skipping")
                try:
                    shutil.rmtree(save_path)
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Failed to save checkpoint to {save_path}: {e}, skipping")
    # def save_ckpt(step=None):
    #     accelerator.wait_for_everyone()
    #     if accelerator.is_main_process:
    #         model=transformer
    #         current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         suffix = f"_{step}" if step is not None else ""
    #         ckpt_dir = f"outputs/smoke_checkpoint_s{2}_{current_time}{suffix}"
    #         os.makedirs(ckpt_dir, exist_ok=True)
    #         unwrapped = accelerator.unwrap_model(model)
    #         with open(os.path.join(ckpt_dir, "pyramid_config.json"), "w") as f:
    #             json.dump(unwrapped.pyramid_config.to_dict(), f, indent=2)
    #         with open(os.path.join(ckpt_dir, "rank_pattern.json"), "w") as f:
    #             json.dump(SMOKE_RANK_PATTERN, f, indent=2)
    #         with open(os.path.join(ckpt_dir, "train_config.json"), "w") as f:
    #             json.dump({"loss_type": args.loss_type}, f, indent=2)
    #         # for n, p in unwrapped.named_parameters():
    #         #     if "norm1" in n:
    #         #         print(n, p.requires_grad, p.shape)
    #         lora_state = get_peft_model_state_dict(unwrapped, adapter_name="default")

    #         # print("LoRA keys count:", len(lora_state))
    #         # for k in lora_state:
    #         #     if 'norm1' in k:
    #         #         print("Found:", k)
                    
    #         StableDiffusion3Pipeline.save_lora_weights(
    #             ckpt_dir, transformer_lora_layers=lora_state,
    #             weight_name="transformer.safetensors")
    #         print(f"Saved: {ckpt_dir}")


    # ======================= 初始化变量 =======================
    global_step = 0
    first_epoch = 0

    # 尝试恢复训练（只读 metadata，不加载权重）
    if args.resume_from_checkpoint:
        global_step, pc, transformer_lora_config, pyramid_loss_type, checkpoint_path = resume_training()
        initial_global_step = global_step
    else:
        initial_global_step = 0
        if args.smoke:
            # smoke: sample_size=16 配128px图
            pc = DEFAULT_PYRAMID_CONFIG
        else:
            pc = DEFAULT_PYRAMID_CONFIG
        transformer_lora_config = None
        pyramid_loss_type = args.loss_type
        checkpoint_path = None

    # 如果恢复时没拿到配置，则用命令行参数构造
    pyramid_loss_type = pyramid_loss_type or args.loss_type
    pc = pc or DEFAULT_PYRAMID_CONFIG
    transformer = TinyPyramidSD3Transformer2DModel.from_flat_pretrained(
        CKPT, pyramid_config=pc,
        subfolder="transformer", revision=args.revision, variant=args.variant,
        torch_dtype=weight_dtype,
    )

    transformer.requires_grad_(False)

    transformer_lora_config = transformer_lora_config or make_lora_config(
        rank_pattern=SMOKE_RANK_PATTERN)
    transformer.add_adapter(transformer_lora_config, adapter_name="default")
    transformer.enable_adapters()

    transformer.to(accelerator.device, dtype=weight_dtype)

    # Cast trainable params (LoRA) to fp32 for FSDP + fp16 mixed precision compatibility
    if args.mixed_precision == "fp16":
        for p in transformer.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    model_parameters = [p for p in transformer.parameters() if p.requires_grad]


    # Optimization parameters
    parameters_with_lr = {"params": model_parameters, "lr": args.learning_rate}
    params_to_optimize = [parameters_with_lr]

    optimizer_class = torch.optim.AdamW
    optimizer_g = optimizer_class(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # optimizer_g = torch.optim.AdamW(model_parameters, lr=args.learning_rate)

    # 数据集与加载器
    if args.smoke:
        from data.data_tiny import SmokeDataset
        train_dataset = SmokeDataset()
    else:
        train_dataset = Real_ESRGAN_Dataset(
            device=accelerator.device, first_stage=True)

    # note: collate_fn 需要访问 accelerator.device，这里用嵌套函数捕获
    def collate_fn(examples):
        latent_stu = torch.stack([e["latent_stu"] for e in examples])
        vae_stu = torch.stack([e["vae_stu"] for e in examples])
        batch = {
            "latent_stu": latent_stu.to(dtype=weight_dtype, device=accelerator.device),
            "vae_stu": vae_stu.to(dtype=weight_dtype, device=accelerator.device),
        }
        if "pooled_prompt_embeds_input" in examples[0]:
            pool = torch.stack([e["pooled_prompt_embeds_input"] for e in examples])
            batch["pool"] = pool.to(dtype=weight_dtype, device=accelerator.device)
        if "latent_stu_teacher_32" in examples[0]:
            teacher_32 = torch.stack([e["latent_stu_teacher_32"] for e in examples])
            batch["latent_stu_teacher_32"] = teacher_32.to(dtype=weight_dtype, device=accelerator.device)
        if "latent_stu_teacher_32_qkv" in examples[0]:
            qkv_batch = [e["latent_stu_teacher_32_qkv"] for e in examples]
            try:
                batch["teacher_q"] = torch.stack([d["q"] for d in qkv_batch]).to(dtype=weight_dtype, device=accelerator.device)
                batch["teacher_k"] = torch.stack([d["k"] for d in qkv_batch]).to(dtype=weight_dtype, device=accelerator.device)
                batch["teacher_v"] = torch.stack([d["v"] for d in qkv_batch]).to(dtype=weight_dtype, device=accelerator.device)
            except KeyError as e:
                logger.warning(f"QKV dict missing key {e}, skipping attn/vr loss for this batch")
                batch.pop("teacher_q", None)
                batch.pop("teacher_k", None)
                batch.pop("teacher_v", None)
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=not args.smoke,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    first_epoch = global_step // num_update_steps_per_epoch

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler_g = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_g,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer_g, train_dataloader, lr_scheduler_g = accelerator.prepare(
        transformer, optimizer_g, train_dataloader, lr_scheduler_g
    )

    # 在 prepare 之后才加载 checkpoint 状态（model/optimizer/scheduler/dataloader 均已注册）
    if checkpoint_path is not None:
        # with torch.no_grad():
        #     unwrapped_model = accelerator.unwrap_model(transformer)
        #     pc = unwrapped_model.pyramid_config

        #     dummy_latent = torch.zeros(
        #         1, pc.in_channels, pc.sample_size, pc.sample_size,
        #         dtype=weight_dtype, device=accelerator.device,
        #     )
        #     dummy_timestep = torch.tensor([DEFAULT_TIMESTEP], device=accelerator.device)
        #     dummy_pool = torch.zeros(
        #         1, pc.pooled_projection_dim,
        #         dtype=weight_dtype, device=accelerator.device,
        #     )

        #     try:
        #         _ = unwrapped_model(
        #             hidden_states=dummy_latent,
        #             timestep=dummy_timestep,
        #             pooled_projections=dummy_pool,
        #             return_dict=False,
        #         )
        #         accelerator.print("🚀 已通过 Dummy Forward 成功激活所有隐式层 Shape！")
        #     except Exception as e:
        #         accelerator.print(f"⚠️ Dummy Forward 失败（如果是缺少某些必填参数请补齐）: {e}")
                
        def load_weight():
            from safetensors.torch import load_file
            weight_path = os.path.join(checkpoint_path,"model.safetensors")
            accelerator.print(f"🔧 正在手动清洗 Checkpoint 权重: {weight_path}")
            state_dict = load_file(weight_path, device="cpu")
            
            unwrapped_model = accelerator.unwrap_model(transformer)
            model_dict = unwrapped_model.state_dict()
            
            # 遍历对比，找出名字相同但 Shape 变了的层
            # 主要是mlp-gate很多size不匹配，当前是0...why
            fixed_state_dict = {}
            for k, v in state_dict.items():
                if k in model_dict:
                    if v.shape != model_dict[k].shape:
                        accelerator.print(f"⚠️ [Shape 冲突已跳过]: 层名 '{k}' | 权重 Shape: {v.shape} -> 模型期望 Shape: {model_dict[k].shape}")
                        continue
                    else:
                        fixed_state_dict[k] = v
                else:
                    fixed_state_dict[k] = v
            unwrapped_model.load_state_dict(fixed_state_dict, strict=False)
            accelerator.print("冲突权重清洗完毕，已成功以 strict=False 强行加载剩余权重！")

        saved_models = accelerator._models
        accelerator._models = []
        accelerator.load_state(checkpoint_path, strict=False)
        accelerator._models = saved_models
        load_weight()

        # 断点续跑时，checkpoint 中保存的 optimizer/scheduler 状态使用的是旧 LR。
        # 必须用新的 --learning_rate 覆盖。注意 scheduler 被 AcceleratedScheduler 包裹，
        # 需通过 .scheduler 访问内部 scheduler 才能修改 base_lrs。
        for pg in optimizer_g.param_groups:
            pg["lr"] = args.learning_rate
        inner_sched = lr_scheduler_g.scheduler if hasattr(lr_scheduler_g, "scheduler") else lr_scheduler_g
        if hasattr(inner_sched, "base_lrs"):
            inner_sched.base_lrs = [args.learning_rate] * len(inner_sched.base_lrs)
            accelerator.print(f"🔧 已用新 LR={args.learning_rate} 覆盖 optimizer 和 scheduler base_lrs")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "tinysr"
        log_name = args.log_name
        time = datetime.datetime.now().strftime('%m-%d_%H:%M')
        wandb_kwargs = {
            "name": f"{log_name}_lr{args.learning_rate}_{time}", "mode": "online"}
        # if args.wandb_id:
        #     if args.wandb_id == "latest":
        #         api = wandb.Api()
        #         entity = args.wandb_entity or ""
        #         runs = api.runs(f"{entity}/{tracker_name}", per_page=1)
        #         if runs:
        #             wandb_kwargs["id"] = runs[0].id
        #             wandb_kwargs["resume"] = "allow"
        #             logger.info(f"Resuming latest wandb run: {runs[0].id}")
        #         else:
        #             logger.warning("No existing wandb runs found. Starting new run.")
        #     elif "/" in args.wandb_id:
        #         api = wandb.Api()
        #         try:
        #             run = api.run(args.wandb_id)
        #             wandb_kwargs["id"] = run.id
        #             wandb_kwargs["resume"] = "allow"
        #             logger.info(f"Resuming wandb run: {run.id}")
        #         except Exception as e:
        #             logger.warning(f"Could not find wandb run {args.wandb_id}: {e}. Starting new run.")
        #     else:
        #         wandb_kwargs["id"] = args.wandb_id
        #         wandb_kwargs["resume"] = "allow"
        wandb_config = vars(args).copy()
        wandb_config["pyramid_config"] = {
            "pyramid_config": pc.to_dict(),
            "lora_r": LORA_R,
            "lora_rank_pattern": SMOKE_RANK_PATTERN,
            "scheme": 2,
            "loss_type": pyramid_loss_type,
            "num_processes": accelerator.num_processes,
        }
        accelerator.init_trackers(tracker_name, config=wandb_config,
                                  init_kwargs={"wandb": wandb_kwargs})
        if args.log_code:
            wandb.run.log_code(".", log_name,
                               include_fn=lambda path: path.endswith(
                                   ".py") or path.endswith(".sh"),
                               exclude_fn=lambda path, root: os.path.relpath(path, root).startswith(".history/"))

    # Train!
    total_batch_size = args.train_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Potentially load in the weights and states from a previous save

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    autocast_ctx = torch.autocast(accelerator.device.type, dtype=weight_dtype)
    # prompt_embeds_default = torch.load("/home/dlw/code/tsdsr/dataset/default/prompt_embeds.pt", map_location=accelerator.device).repeat(args.train_batch_size, 1, 1)
    if not args.smoke:
        pooled_prompt_embeds_default = torch.load(
            "dataset/default/pool_embeds.pt", map_location=accelerator.device)
            
    # ================================================================
    # 📊 多卡安全版：验证 [optimizer_g] 与 [lr_scheduler_g] 状态审计
    # ================================================================
    # 只有主进程（Rank 0）负责打印审计日志，防止多卡刷屏
    if checkpoint_path is not None and accelerator.is_main_process:
        accelerator.print("\n" + "="*50)
        accelerator.print("🔍 === [多卡分布式：Scheduler & Optimizer 恢复状态审计] ===")
        accelerator.print("="*50)
        
        current_internal_step = getattr(lr_scheduler_g, "last_epoch", "未找到计数器")
        accelerator.print(f"📈 [Scheduler] 内部当前 Step 计数 (last_epoch): {current_internal_step}")
        
        for i, param_group in enumerate(optimizer_g.param_groups):
            current_lr = param_group.get("lr", 0.0)
            accelerator.print(f"  👉 参数组 [{i}]: 当前实际运行 LR = {current_lr:<12}")
            
        if len(optimizer_g.state) > 0:
            sample_param = list(optimizer_g.state.keys())[0]
            param_state = optimizer_g.state[sample_param]
            if "exp_avg" in param_state:
                m_mean = param_state["exp_avg"].mean().item()
                accelerator.print(f"  |- 一阶动量指纹均值 (exp_avg mean): {m_mean:.8f}")
            if "step" in param_state:
                p_step = param_state["step"]
                p_step_val = p_step.item() if hasattr(p_step, "item") else p_step
                accelerator.print(f"  |- 优化器底层参数级 Step 计数:   {p_step_val}")
        else:
            accelerator.print("  🚨 零度冻结警告：多卡优化器状态为空！")
        accelerator.print("="*50 + "\n")

    # ================================================================
    # ⚡ 多卡广播唤醒：修复多卡下 Scheduler 锁死 LR=0 的 Bug
    # ================================================================
    # 所有人（所有卡）都要参与这个逻辑，确保步数同步
    real_current_step = global_step  
    
    # 所有人同步修改内部计数
    if hasattr(lr_scheduler_g, "last_epoch"):
        lr_scheduler_g.last_epoch = real_current_step
    elif hasattr(lr_scheduler_g, "_step_count"):
        lr_scheduler_g._step_count = real_current_step
        
    try:
        # 所有卡一起执行 step() 刷新各自对应的优化器学习率
        lr_scheduler_g.step()
    except Exception as e:
        if accelerator.is_main_process:
            accelerator.print(f"❌ 多卡强制唤醒调度器失败: {e}")

    # 最后由主进程打印最终确认结果
    if accelerator.is_main_process:
        accelerator.print(f"重新校准后的实际运行 LR 已变为: {optimizer_g.param_groups[0]['lr']}\n")
    

    accelerator.wait_for_everyone()
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # 根据当前 batch 的实际大小 repeat embedding
                current_bs = batch["vae_stu"].shape[0]
                if "pool" in batch:
                    pool_emb = batch["pool"]
                else:
                    pool_emb = pooled_prompt_embeds_default.repeat(current_bs, 1)

                with autocast_ctx:
                    latent_teacher = batch["latent_stu"]
                    # latent_teacher = batch["latent_hr"]
                    model_input = batch["vae_stu"]
                    timesteps = torch.tensor(
                        [DEFAULT_TIMESTEP], device=accelerator.device)

                    out, pre_last, h_after_bridge = transformer(
                        hidden_states=model_input,
                        timestep=timesteps,
                        pooled_projections=pool_emb,
                        return_dict=False,
                    )
                    if pyramid_loss_type == "a":
                        scale_factor = 64 // pc.sample_size
                        # scale_factor = pc.p_states[-1].grid_hw // pc.p_states[0].grid_hw
                        input_up = F.interpolate(model_input.float(
                        ), scale_factor=scale_factor, mode='bilinear', align_corners=False)
                        denoised = input_up - out.float()
                        loss_g = F.l1_loss(
                            denoised, latent_teacher.float().detach())
                    else:
                        last_grid_hw = pc.p_states[-1].grid_hw
                        latent_before = transformer._tokens_to_latent(
                            pre_last.float(), last_grid_hw)
                        denoised = latent_before.float() - out.float()
                        loss_g = F.l1_loss(
                            denoised, latent_teacher.float().detach())
                    loss_l1 = loss_g.detach().item()


                    # hidden cosine_similarity
                    cos_loss = 0.0
                    if args.cos_loss_weight > 0 and "latent_stu_teacher_32" in batch:
                        unwrapped_for_cos = accelerator.unwrap_model(transformer)
                        if hasattr(unwrapped_for_cos, '_distill_h') and unwrapped_for_cos._distill_h is not None:
                            teacher_32 = batch["latent_stu_teacher_32"].float()
                            student_h = unwrapped_for_cos._distill_h.float()
                            cos_sim = F.cosine_similarity(student_h, teacher_32, dim=-1)

                            loss_cos = args.cos_loss_weight * (1.0 - cos_sim.mean())
                            loss_g = loss_g + loss_cos
                            cos_loss = loss_cos.detach().item()


                    attn_loss = 0.0
                    vr_loss = 0.0
                    need_distill = (args.attn_loss_weight > 0 and "teacher_q" in batch) \
                        or (args.vr_loss_weight > 0 and "teacher_v" in batch) \
                        or (args.cos_qkv_loss_weight > 0 and "teacher_q" in batch)
                    if need_distill:
                        unwrapped = accelerator.unwrap_model(transformer)
                    else:
                        unwrapped = None

                    if need_distill and args.attn_loss_weight > 0 \
                            and hasattr(unwrapped, '_distill_q') and unwrapped._distill_q is not None:
                        teacher_q = batch["teacher_q"].float()
                        teacher_k = batch["teacher_k"].float()
                        student_q = unwrapped._distill_q.float()
                        student_k = unwrapped._distill_k.float()
                        head_dim = teacher_q.shape[-1]
                        scale = 1.0 / (head_dim ** 0.5)
                        tau = args.attn_temperature
                        num_heads = teacher_q.shape[1]
                        chunk = min(4, num_heads)
                        loss_at = 0.0
                        for h in range(0, num_heads, chunk):
                            h_end = min(h + chunk, num_heads)
                            t_q = teacher_q[:, h:h_end]
                            t_k = teacher_k[:, h:h_end]
                            s_q = student_q[:, h:h_end]
                            s_k = student_k[:, h:h_end]
                            t_log = F.log_softmax(
                                torch.matmul(t_q, t_k.transpose(-1, -2)) * scale / tau, dim=-1)
                            s_log = F.log_softmax(
                                torch.matmul(s_q, s_k.transpose(-1, -2)) * scale / tau, dim=-1)
                            kl = F.kl_div(s_log, t_log, reduction='none', log_target=True)
                            loss_at += kl.sum(dim=-1).mean() \
                                * (h_end - h) / num_heads
                        loss_at = args.attn_loss_weight * loss_at
                        loss_g = loss_g + loss_at
                        attn_loss = loss_at.detach().item()

                    if need_distill and args.vr_loss_weight > 0 \
                            and hasattr(unwrapped, '_distill_v') and unwrapped._distill_v is not None:
                        teacher_v = batch["teacher_v"].float()
                        student_v = unwrapped._distill_v.float()
                        head_dim = teacher_v.shape[-1]
                        scale = 1.0 / (head_dim ** 0.5)
                        tau = args.attn_temperature
                        num_heads = teacher_v.shape[1]
                        chunk = min(4, num_heads)
                        loss_vr = 0.0
                        for h in range(0, num_heads, chunk):
                            h_end = min(h + chunk, num_heads)
                            t_v = teacher_v[:, h:h_end]
                            s_v = student_v[:, h:h_end]
                            t_log = F.log_softmax(
                                torch.matmul(t_v, t_v.transpose(-1, -2)) * scale / tau, dim=-1)
                            s_log = F.log_softmax(
                                torch.matmul(s_v, s_v.transpose(-1, -2)) * scale / tau, dim=-1)
                            kl = F.kl_div(s_log, t_log, reduction='none', log_target=True)
                            loss_vr += kl.sum(dim=-1).mean() \
                                * (h_end - h) / num_heads
                        loss_vr = args.vr_loss_weight * loss_vr
                        loss_g = loss_g + loss_vr
                        vr_loss = loss_vr.detach().item()

                    cos_qkv_loss = 0.0
                    if args.cos_qkv_loss_weight > 0 \
                            and hasattr(unwrapped, '_distill_q') and unwrapped._distill_q is not None \
                            and 'teacher_q' in batch:
                        teacher_q = batch["teacher_q"].float()
                        teacher_k = batch["teacher_k"].float()
                        teacher_v = batch["teacher_v"].float()
                        student_q = unwrapped._distill_q.float()
                        student_k = unwrapped._distill_k.float()
                        student_v = unwrapped._distill_v.float()
                        loss_q = (1.0 - F.cosine_similarity(student_q, teacher_q, dim=-1)).mean()
                        loss_k = (1.0 - F.cosine_similarity(student_k, teacher_k, dim=-1)).mean()
                        loss_v = (1.0 - F.cosine_similarity(student_v, teacher_v, dim=-1)).mean()
                        loss_cos_qkv = args.cos_qkv_loss_weight * (loss_q + loss_k + loss_v) / 3.0
                        loss_g = loss_g + loss_cos_qkv
                        cos_qkv_loss = loss_cos_qkv.detach().item()

                if global_step % 10 == 0 or global_step == initial_global_step:
                    accelerator.print(
                        f"[LOSS DEBUG] step={global_step} "
                        f"l1={loss_l1:.6f} "
                        f"cos={cos_loss:.6f}(w={args.cos_loss_weight}) "
                        f"attn={attn_loss:.6f}(w={args.attn_loss_weight}) "
                        f"vr_raw={(vr_loss/(args.vr_loss_weight or 1.0)):.6f} vr_weighted={vr_loss:.6f}(w={args.vr_loss_weight}) "
                        f"cos_qkv={cos_qkv_loss:.6f}(w={args.cos_qkv_loss_weight}) "
                        f"total_recorded={loss_g.detach().item():.6f} "
                        f"total_expected={loss_l1 + (cos_loss if args.cos_loss_weight > 0 else 0) + (attn_loss if args.attn_loss_weight > 0 else 0) + vr_loss + (cos_qkv_loss if args.cos_qkv_loss_weight > 0 else 0):.6f} "
                        f"need_distill={need_distill} "
                        f"has_tv={'teacher_v' in batch} "
                        f"has_tq={'teacher_q' in batch} "
                        f"has_t32={'latent_stu_teacher_32' in batch}"
                    )

                # backward
                accelerator.backward(loss_g)


                if accelerator.sync_gradients:
                    params_to_clip = model_parameters
                    accelerator.clip_grad_norm_(
                        params_to_clip, args.max_grad_norm)
                optimizer_g.step()
                lr_scheduler_g.step()
                optimizer_g.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.checkpointing_steps == 1:
                    print("save_ckpt")
                    save_ckpt(step=global_step)

                logs = {"l1_loss": loss_l1, "loss_total": loss_g.detach().item(), "lr": optimizer_g.param_groups[0]["lr"],
                        "cos_loss": cos_loss/(args.cos_loss_weight or 1.0),
                        "attn_loss": attn_loss/(args.attn_loss_weight or 1.0),
                        "vr_loss": vr_loss/(args.vr_loss_weight or 1.0),
                        "cos_qkv_loss": cos_qkv_loss/(args.cos_qkv_loss_weight or 1.0)}
                progress_bar.set_postfix(**logs)
                if accelerator.is_main_process:
                    accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    save_ckpt()  # 最终保存
    if accelerator.is_main_process:
        wandb.finish()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

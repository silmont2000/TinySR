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
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
import timm
from data.data_tiny import Real_ESRGAN_Dataset
import diffusers
from diffusers import (
    StableDiffusion3Pipeline,
)
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.tinysr.sd3 import SD3Transformer2DModel
from models.tinysr.pyramid_config import PyramidArchConfig
from models.tinysr.pyramid_model import TinyPyramidSD3Transformer2DModel
from models.tinysr.stage1_defaults import (
    CKPT, SMOKE_RANK_PATTERN, make_lora_config,
    LORA_R, DEFAULT_TIMESTEP, VAE_CKPT, PYRAMID_MULT_CONFIG,
)

from diffusers.image_processor import  VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module
from models.vae.autoencoder_tiny import AutoencoderTiny
from models.vae.autoencoder_kl import AutoencoderKL
import utils.util_net as util_net
from models.discriminator import ProjectedDiscriminator
from utils.util import load_lora_state_dict, load_lora_state_dict_warn
from models.vit import vit_small, vit_large
if is_wandb_available():
    import wandb
import pyiqa
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
        f"Running validation... \n Generating images with prompt:"
    )

    images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)
        
    with autocast_ctx:
        hq = torch.chunk(log_dict["hq"], args.train_batch_size, dim=0)[0]
        lq = torch.chunk(log_dict["lq"], args.train_batch_size, dim=0)[0]
        image_stu = torch.chunk(log_dict["image_stu"], args.train_batch_size, dim=0)[0]

        image_processor = VaeImageProcessor(vae_scale_factor=2 ** (vae_scale - 1))
        image = image_processor.postprocess(hq.detach())[0]
        images.append(image)
        image = image_processor.postprocess(lq.detach())[0]
        images.append(image)
        image = image_processor.postprocess(image_stu.detach())[0]
        images.append(image)
        
    caption=["hq", "lq", "image_stu"]
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
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # parser.add_argument(
    #     "--pretrained_model_name_or_path",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="Path to pretrained model or model identifier from huggingface.co/models.",
    # )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--DINO_v2_pretrained_model_path",
        type=str,
        default=None,
        required=True,
        help="Path to DINOv2 pretrained model.",
    )
    # parser.add_argument(
    #     "--teacher_model_name_or_path",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="Path to DINOv2 pretrained model.",
    # )
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
        "--output_dir",
        type=str,
        default="tsdsr-checkpoint",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--log_name",
        type=str,
        default="tsdsr",
        help="log_name",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--log_code",
        action="store_true",
        help="log code",
    )
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
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=5000,
        help="Number of steps to log validation images.",
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
        "--learning_rate_fake",
        type=float,
        default=1e-6,
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
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
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
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
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
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
    )
    parser.add_argument("--loss_type", type=str, default="a", choices=["a", "b"],
                        help="Pyramid loss type (a=simple, b=multi-scale)")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def collate_fn(examples, weight_dtype=torch.float16):
    lr_img = [example["lr_img"] for example in examples]
    hr_img = [example["hr_img"] for example in examples]
    # latent_hr = [example["latent_hr"] for example in examples]
    vae_stu = torch.stack([e["vae_stu"] for e in examples])
    
    # prompts = [example["prompt_text"] for example in examples]
    # prompt_embeds = torch.stack([example["prompt_embeds_input"] for example in examples])
    pooled_prompt_embeds = torch.stack([example["pooled_prompt_embeds_input"] for example in examples])

    latent_stu = [example["latent_stu"] for example in examples]
    latent_stu = torch.stack(latent_stu)

    lr_img = torch.stack(lr_img)
    hr_img = torch.stack(hr_img)
    # latent_hr = torch.stack(latent_hr)
    
    batch = {
        "lr_img": lr_img.to(dtype=weight_dtype),
        "hr_img": hr_img.to(dtype=weight_dtype),
        # "latent_hr": latent_hr.to(dtype=weight_dtype),
        "latent_stu": latent_stu.to(dtype=weight_dtype),
        "vae_stu": vae_stu.to(dtype=weight_dtype),

        # "prompts": prompts,
        # "prompt_embeds": prompt_embeds.to(dtype=weight_dtype),
        
        "pooled_prompt_embeds": pooled_prompt_embeds.to(dtype=weight_dtype),
             }
    return batch

def main(args):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
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
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

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
    
    weight_dtype = torch.float16
    if accelerator.mixed_precision == "no":
        weight_dtype = torch.float32
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    # transformer = TinySD3Transformer2DModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    # )
    pc_path = os.path.join(args.lora_dir, "pyramid_config.json")
    if not os.path.exists(pc_path):
        raise FileNotFoundError(f"pyramid_config.json not found in {args.lora_dir}")
    with open(pc_path) as f:
        pc = PyramidArchConfig.from_dict(json.load(f))
    mult_config_path = None
    candidate = os.path.join(args.lora_dir, "mult_config.json")
    if os.path.isfile(candidate):
        mult_config_path = candidate
        print(f"  mult_config loaded from {mult_config_path}")
    transformer = TinyPyramidSD3Transformer2DModel.from_flat_pretrained(
        CKPT, pyramid_config=pc,
        subfolder="transformer",
        revision=args.revision, variant=args.variant,
        torch_dtype=weight_dtype,
        ignore_mismatched_sizes=True,
        mult_config_path=mult_config_path,
    )
    vae = AutoencoderTiny.from_pretrained("checkpoint/vae/separable")
    vae_decode = AutoencoderKL.from_pretrained("/data/disk2/xby/sd3-medium", subfolder="vae").to("cuda", weight_dtype)


    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # transformer_lora_config = LoraConfig(
    #     r=64,
    #     lora_alpha=64,
    #     init_lora_weights="gaussian",
    #     target_modules=["to_k", "to_q", "to_v", "to_out.0","proj","linear", "linear_1", "linear_2", "net.2"],
    # )
    rp_path = os.path.join(args.lora_dir, "rank_pattern.json")
    if os.path.exists(rp_path):
        with open(rp_path) as f:
            rank_pattern = json.load(f)
    else:
        rank_pattern = SMOKE_RANK_PATTERN
    transformer_lora_config = make_lora_config(rank_pattern=rank_pattern)
    transformer.add_adapter(transformer_lora_config, adapter_name="default")
    transformer.enable_adapters()
    transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(args.lora_dir, weight_name="model.safetensors")
    load_lora_state_dict_warn(transformer_lora_state_dict, transformer)

    
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            for index, model in enumerate(models):
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model, adapter_name="default")
                    StableDiffusion3Pipeline.save_lora_weights(
                output_dir, transformer_lora_layers=transformer_lora_layers_to_save,weight_name=f"model.safetensors"
            ) 
                elif isinstance(model, type(unwrap_model(model_dis))) :
                    torch.save(model.state_dict(), os.path.join(output_dir, f"model_dis.safetensors"))
                else:
                    pass
                weights.pop()

    def load_model_hook(models, input_dir):
        model_dis = models.pop()
        transformer = models.pop()
        transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir,weight_name="model.safetensors")
        load_lora_state_dict(transformer_lora_state_dict, transformer)
        model_dis.load_state_dict(torch.load(os.path.join(input_dir, "model_dis.safetensors")))

        if args.mixed_precision == "fp16":
            models = [transformer, model_dis]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    model_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters())) 
    if accelerator.is_main_process:
        for name, param in transformer.named_parameters():
            if param.requires_grad:
                logger.info(f"Trainable parameter: {name}")

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        cast_training_params(models, dtype=torch.float32)

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

    model_fea = vit_large(patch_size=14, img_size=518, block_chunks=0, init_values=1.0,num_register_tokens=4)
    util_net.reload_model(model_fea, torch.load(args.DINO_v2_pretrained_model_path))
    patch_resolution = 16 * (512 // 256)
    model_fea.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
        model_fea.pos_embed.data, [patch_resolution, patch_resolution],
    )
    model_fea.requires_grad_(False)
    model_fea.to(accelerator.device, dtype=weight_dtype)

    model_dis = ProjectedDiscriminator(c_dim=1024).train()
    optimizer_Dis = torch.optim.AdamW(
        model_dis.parameters(),
        lr=args.learning_rate_discrimitor,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = Real_ESRGAN_Dataset(device=accelerator.device)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, weight_dtype),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
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
    if getattr(accelerator.state, "fsdp_plugin", None):
        fsdp_plugin = accelerator.state.fsdp_plugin
        fsdp_plugin.ignored_modules = [vae]
        
    # Prepare everything with our `accelerator`.
    transformer, model_dis = accelerator.prepare(transformer, model_dis)
    train_dataloader, optimizer_g, lr_scheduler_g, optimizer_Dis  = accelerator.prepare(train_dataloader, optimizer_g, lr_scheduler_g, optimizer_Dis)        

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "tinysr-stage2"
        log_name = args.log_name
        time = datetime.datetime.now().strftime('%m-%d_%H:%M')
        accelerator.init_trackers(tracker_name, config=vars(args), 
                                  init_kwargs={"wandb": {"name": f"{log_name}_lr{args.learning_rate}_{time}","mode": 'online'}}
                                  )
        if args.log_code:
            wandb.run.log_code(".", log_name,
                           include_fn=lambda path: path.endswith(".py") or path.endswith(".sh"),
                           exclude_fn=lambda path, root: os.path.relpath(path, root).startswith(".history/"))
        
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            if "latest-checkpoint" in dirs:
                path  = "latest-checkpoint"
            else:
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        elif path == "latest-checkpoint":
            accelerator.print(f"Resuming from checkpoint {path} and starting a new training run.")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = 0

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    log_dict = {}
    lpips = pyiqa.create_metric('lpips', as_loss=True).cuda()
    autocast_ctx = torch.autocast(accelerator.device.type,dtype=weight_dtype)
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        model_dis.train()
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer, model_dis]
            with accelerator.accumulate(models_to_accumulate):
                lr_values = batch["lr_img"]
                hr_values = batch["hr_img"]
                pooled_prompt_embeds = batch["pooled_prompt_embeds"]
                log_dict["lq"] = lr_values.float().cpu()
                log_dict["hq"] = hr_values.float().cpu()
                with autocast_ctx:
                    # with torch.no_grad():
                    #     model_input = vae.encode(lr_values).latents * vae.config.scaling_factor
                    #     timesteps = torch.tensor([1000.], device=accelerator.device)
                    #     latent_tea = batch["latent_stu"]
                    timesteps = torch.tensor([DEFAULT_TIMESTEP], device=accelerator.device)
                    model_input = batch["vae_stu"]
                    latent_tea = batch["latent_stu"]
                       
                    model_pred, _,_ = transformer(
                        hidden_states=model_input,
                        timestep=timesteps,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )
                    scale_factor = 64 // pc.sample_size

                    # scale_factor = DEFAULT_PYRAMID_CONFIG.p_states[-1].grid_hw // DEFAULT_PYRAMID_CONFIG.p_states[0].grid_hw
                    input_up = F.interpolate(model_input.float(
                    ), scale_factor=scale_factor, mode='bilinear', align_corners=False)
                    latent_stu = input_up - model_pred.float()

                    l1_loss =  F.l1_loss(latent_stu.float(), latent_tea.float().detach(), reduction='mean') 
                    
                    image_stu = vae.decode(latent_stu / vae.config.scaling_factor, return_dict=False)[0].clamp(-1, 1)
                    if accelerator.is_main_process and step % args.validation_steps == 49:
                        log_dict["image_stu"] = image_stu.cpu()

                    image_stu = (image_stu * 0.5 + 0.5)
                    hr_values = (hr_values * 0.5 + 0.5)

                    # discrimitor
                    _, cls_lr = model_fea(F.interpolate(hr_values, size=448, mode='bicubic'))
                    pred_real, features = model_dis(hr_values, cls_lr.detach())
                    pred_fake, _ = model_dis(image_stu.detach(), cls_lr.detach()) 
                    pred_fake = torch.cat(pred_fake, dim=1)
                    loss_fake = torch.mean(torch.relu(1.0 + pred_fake))
                    pred_real = torch.cat(pred_real, dim=1)
                    loss_real = torch.mean(torch.relu(1.0 - pred_real))
                    loss_D = loss_real + loss_fake
                    # compute the generator loss
                    pred_fake, _ = model_dis(image_stu, cls_lr.detach())
                    pred_fake = torch.cat(pred_fake, dim=1)
                    gan_loss = -torch.mean(pred_fake)

                    lpips_loss = lpips(image_stu, hr_values) 
                    # Compute total loss
                    loss_g = 0.3 * gan_loss  + 1 * lpips_loss + 5 * l1_loss
                    # loss_g = 0.3 * gan_loss  + 1 * lpips_loss
                    # loss_g = 0.3 * gan_loss

                # backward
                accelerator.backward(loss_g)
                if accelerator.sync_gradients:
                    params_to_clip = model_parameters
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_g.step()
                lr_scheduler_g.step()
                optimizer_g.zero_grad()
                
                accelerator.backward(loss_D)
                optimizer_Dis.step()
                optimizer_Dis.zero_grad(set_to_none=args.set_grads_to_none)
                model_fea.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 500:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        try:
                            os.makedirs(save_path, exist_ok=True)
                            unwrapped = accelerator.unwrap_model(transformer)
                            with open(os.path.join(save_path, "mult_config.json"), "w") as f:
                                json.dump(unwrapped._mult_config, f, indent=2)
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
                        except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
                            logger.warning(f"Memory insufficient, skipping checkpoint save at step {global_step}: {e}")
                            if os.path.exists(save_path):
                                shutil.rmtree(save_path)
                            torch.cuda.empty_cache()

            logs = {
                    "lpips_loss": lpips_loss.detach().item(), 
                    "gan_loss": gan_loss.detach().item(),
                    "discrimitor_loss": loss_D.detach().item(),
                    "l1_loss": l1_loss.detach().item(),
                    }
            progress_bar.set_postfix(**logs)
            if accelerator.is_main_process:
                accelerator.log(logs, step=global_step)
            
            if global_step >= args.max_train_steps:
                break
            
            if accelerator.is_main_process and step % args.validation_steps == 49:
                log_validation(log_dict, args,accelerator, len(unwrap_model(vae).config.block_out_channels))

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if accelerator.sync_gradients:
            save_path = os.path.join(args.output_dir, f"checkpoint-latest")
            try:
                os.makedirs(save_path, exist_ok=True)
                with open(os.path.join(save_path, "mult_config.json"), "w") as f:
                    json.dump(accelerator.unwrap_model(transformer)._mult_config, f, indent=2)
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
            except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
                logger.warning(f"Memory insufficient, skipping final checkpoint save: {e}")
                if os.path.exists(save_path):
                    shutil.rmtree(save_path)
                torch.cuda.empty_cache()

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)
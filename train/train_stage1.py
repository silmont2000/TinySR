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
from data.data_tiny import Real_ESRGAN_Dataset
import diffusers
from diffusers import (
    StableDiffusion3Pipeline,
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
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.tinysr.sd3 import SD3Transformer2DModel
from models.tinysr.pyramid_config import PyramidArchConfig, PStateSpec
from models.tinysr.pyramid_model import TinyPyramidSD3Transformer2DModel
from utils.util import load_lora_state_dict_warn
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
        required=True,
        help="Path to DINOv2 pretrained model.",
    )
    parser.add_argument(
        "--use_pyramid",
        action="store_true",
        default=False,
        help="Use TinyPyramidSD3Transformer2DModel instead of flat TinySD3Transformer2DModel.",
    )
    parser.add_argument(
        "--pyramid_num_blocks",
        type=int,
        nargs="+",
        default=[4, 4, 4],
        help="Number of blocks per pyramid state.",
    )
    parser.add_argument(
        "--pyramid_dims",
        type=int,
        nargs="+",
        default=[1536, 1536, 1536],
        help="Hidden dimension per pyramid state.",
    )
    parser.add_argument(
        "--pyramid_grid_hw",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="Token grid HW per pyramid state.",
    )
    parser.add_argument(
        "--pyramid_sample_size",
        type=int,
        default=64,
        help="VAE latent spatial size (64 for 512px input, 16 for 128px input).",
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
        help="wandb run id to resume (e.g. 9at5m8zu).",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--log_code",action="store_true",help="log code",)
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
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
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

def collate_fn(examples, weight_dtype=torch.float16):
    
    latent_stu = [example["latent_stu"] for example in examples]
    vae_stu = [example["vae_stu"] for example in examples]

    latent_stu = torch.stack(latent_stu)
    vae_stu = torch.stack(vae_stu)

    
    batch = {
        "latent_stu": latent_stu.to(dtype=weight_dtype, device="cuda"),
        "vae_stu": vae_stu.to(dtype=weight_dtype, device="cuda"),
             }
    if "target_8" in examples[0]:
        batch["target_8"] = torch.stack([e["target_8"] for e in examples]).to(dtype=weight_dtype, device="cuda")
        batch["target_16"] = torch.stack([e["target_16"] for e in examples]).to(dtype=weight_dtype, device="cuda")
        batch["target_32"] = torch.stack([e["target_32"] for e in examples]).to(dtype=weight_dtype, device="cuda")
    return batch

def main(args):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
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

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float16
    if accelerator.mixed_precision == "no":
        weight_dtype = torch.float32
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler and models
    if args.use_pyramid:
        pc = PyramidArchConfig(
            p_states=tuple(
                PStateSpec(n, d, g)
                for n, d, g in zip(args.pyramid_num_blocks, args.pyramid_dims, args.pyramid_grid_hw)
            ),
            sample_size=args.pyramid_sample_size,
        )
        transformer = TinyPyramidSD3Transformer2DModel.from_flat_pretrained(
            args.pretrained_model_name_or_path, pyramid_config=pc,
            subfolder="transformer", revision=args.revision, variant=args.variant,
            torch_dtype=weight_dtype,
        )
    else:
        transformer = TinySD3Transformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )
    transformer.requires_grad_(False)
    
    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    transformer_lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0","proj","linear", "linear_1", "linear_2", "net.2"],
        rank_pattern={
            "p_states.0": 16,
            "p_states.1": 64,
            "p_states.2": 256,
        },
    )
    transformer.add_adapter(transformer_lora_config, adapter_name="default")
    transformer.enable_adapters()
    
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
                output_dir, transformer_lora_layers=transformer_lora_layers_to_save,weight_name=f"transformer.safetensors"
            ) 
                else:
                    pass
                weights.pop()

    def load_model_hook(models, input_dir):
        transformer = models.pop()
        
        transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir, weight_name="transformer.safetensors")
        load_lora_state_dict_warn(transformer_lora_state_dict, transformer)
        
        if args.mixed_precision == "fp16":
            models = [transformer]
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

    # Dataset and DataLoaders creation:
    train_dataset = Real_ESRGAN_Dataset(device=accelerator.device, multi_stage=args.use_pyramid)
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
        
    transformer  = accelerator.prepare(transformer)
    optimizer_g, train_dataloader, lr_scheduler_g = accelerator.prepare(optimizer_g, train_dataloader, lr_scheduler_g)        

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "tinysr"
        log_name = args.log_name
        time = datetime.datetime.now().strftime('%m-%d_%H:%M')
        wandb_kwargs = {"name": f"{log_name}_lr{args.learning_rate}_{time}", "mode": "online"}
        if args.wandb_id:
            wandb_kwargs["id"] = args.wandb_id
            wandb_kwargs["resume"] = "allow"
        wandb_config = vars(args).copy()
        if args.use_pyramid:
            wandb_config["pyramid_config"] = {
                "num_blocks": args.pyramid_num_blocks,
                "dims": args.pyramid_dims,
                "grid_hw": args.pyramid_grid_hw,
                "sample_size": args.pyramid_sample_size,
            }
            wandb_config["lora_rank_pattern"] = {
                "p_states.0": 16, "p_states.1": 64, "p_states.2": 256,
            }
        accelerator.init_trackers(tracker_name, config=wandb_config,
                                  init_kwargs={"wandb": wandb_kwargs})
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
            accelerator.print(f"Resuming from checkpoint {path}")
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
    
    autocast_ctx = torch.autocast(accelerator.device.type,dtype=weight_dtype)
    # prompt_embeds_default = torch.load("/home/dlw/code/tsdsr/dataset/default/prompt_embeds.pt", map_location=accelerator.device).repeat(args.train_batch_size, 1, 1)
    pooled_prompt_embeds_default = torch.load("dataset/default/pool_embeds.pt", map_location=accelerator.device).repeat(args.train_batch_size, 1)
    
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                with autocast_ctx:
                    latent_teacher = batch["latent_stu"]
                    model_input = batch["vae_stu"]
                    timesteps = torch.tensor([1000.], device=accelerator.device)
                    
                    if args.use_pyramid:
                        output, stage_noises, stage_inputs = transformer(
                            hidden_states=model_input,
                            timestep=timesteps,
                            pooled_projections=pooled_prompt_embeds_default,
                            return_dict=False,
                        )
                        mi_f = model_input.to(output.dtype)
                        loss0 = F.l1_loss((mi_f - stage_noises[0]).float(),
                                          batch["target_8"].float().detach(), reduction='mean')
                        loss1 = F.l1_loss((stage_inputs[1] - stage_noises[1]).float(),
                                          batch["target_16"].float().detach(), reduction='mean')
                        loss2 = F.l1_loss((stage_inputs[2] - out).float(),
                                          batch["target_32"].float().detach(), reduction='mean')
                        loss_g = 0.1 * loss0 + 0.3 * loss1 + 1.0 * loss2
                    else:
                        model_pred = transformer(
                            hidden_states=model_input,
                            timestep=timesteps,
                            pooled_projections=pooled_prompt_embeds_default,
                            return_dict=False,
                        )[0]
                        latent_stu =  model_input - model_pred 
                        loss_g = 1 * F.l1_loss(latent_stu.float(), latent_teacher.detach().float(), reduction='mean')
                    
                # backward
                accelerator.backward(loss_g)
                if accelerator.sync_gradients:
                    params_to_clip = model_parameters
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_g.step()
                lr_scheduler_g.step()
                optimizer_g.zero_grad()

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
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                    "l1_loss": l1_loss.detach().item(),
                    }
            progress_bar.set_postfix(**logs)
            if accelerator.is_main_process:
                accelerator.log(logs, step=global_step)
            
            if global_step >= args.max_train_steps:
                break
            
    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if accelerator.sync_gradients:
            save_path = os.path.join(args.output_dir, f"checkpoint-latest")
            accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
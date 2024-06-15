# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main training loop."""

import os, clip
import time
import copy, random
import numpy as np
import torch
import dnnlib
from torch_utils.misc import load_checkpoint
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from training.evaluate import save_image_grid, save_video_grid

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as NativeDDP
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
except ImportError:
    pass
from timm.utils import ApexScaler

from datasets.human_promptset import HumanPrompts as TextsPrompts

from .generator import TriPlaneGenerator
from training.camera_pose_sampling import NeRFDataset
from training.loss import RenderingLoss


def preprocess_texts(t):
    while(True):
        if t.startswith('"'):
            t = t[1:]
        else:
            break

    while(True):
        if t.endswith('.') or t.endswith(','):
            t = t[:-1]
        else:
            break
    return t + '.'


def training_loop(rank, args):
    num_gpus = args.num_gpus
    random_seed = args.random_seed
    batch_size = args.batch_size
    rendering_h = rendering_w = args.neural_rendering_resolution_initial
    pin_memory = args.pin_memory
    # Initialize.
    # device = torch.device('cuda', rank) # 单机多卡
    device = torch.device('cuda') # 多机多卡
    np.random.seed(random_seed * num_gpus + rank)
    random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.cuda.manual_seed(random_seed * num_gpus + rank)
    torch.cuda.manual_seed_all(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark    # Improves training speed.
    # torch.backends.cudnn.deterministic = True    # For reproducibility
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed. # TODO: ENABLE
    grid_sample_gradfix.enabled = False                  # Avoids errors with the augmentation pipe.

    def _init_fn(worker_id):
        np.random.seed(random_seed * num_gpus + rank)

    valid_pose_num = 121

    if rank == 0:
        print('Loading testing prompt set...')
        test_prompt_set = TextsPrompts(start=0.0, end=1.0)
        valid_prompt_loader = torch.utils.data.DataLoader(dataset=test_prompt_set, sampler=None, shuffle=False, batch_size=3,
                                                         num_workers=args.num_workers, pin_memory=pin_memory, prefetch_factor=2, worker_init_fn=_init_fn)

    if rank == 0:
        print('Loading camera pose sampling datalodader for testing...')
        valid_pose_loader = NeRFDataset(args, type='val', H=args.resolution_testing, W=args.resolution_testing, size=valid_pose_num).dataloader(shuffle=False,
                                               pin_memory=False, worker_init_fn=_init_fn, num_workers=args.num_workers) # * batch_size // num_gpus

    # Construct Triplane
    G = TriPlaneGenerator(args.rendering_kwargs, density_activation=args.density_activation, guidance=args.guidance)
    G.to(device)

    # Print network summary tables.
    if rank == 0:
        valid_prompts_imgs = []

        for prompt in valid_prompt_loader:
            valid_prompts_imgs.extend(prompt)

        grid_size = (valid_pose_num, len(test_prompt_set))
        valid_prompts_planes = valid_prompts_imgs

        print('Saving validation prompts...')
        valid_prompts = [prompt + '\n' for prompt in valid_prompts_planes]
        with open(os.path.join(args.run_dir, 'valid_prompts.txt'), 'w') as f:
            f.writelines(valid_prompts)

    loss = RenderingLoss(device=device, rank=rank, amp_train=args.amp_train, amp_level=args.amp_level, guid_name=args.guidance, **args.loss_kwargs,
                         pretrained_path=args.pretrained_path)
    torch.cuda.empty_cache()
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, args.opt_kwargs, args.reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer

            if args.amp_train:
                module, opt = amp.initialize(module, opt, opt_level=args.amp_level)
                loss_scaler_both = ApexScaler()
            else:
                loss_scaler_both = None
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            if args.amp_train:
                module, opt = amp.initialize(module, opt, opt_level=args.amp_level)
                loss_scaler_main = ApexScaler()
                loss_scaler_reg = ApexScaler()
            else:
                loss_scaler_main = None
                loss_scaler_reg = None

        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        G_ema = copy.deepcopy(module)
        G_ema.update_resolution(args.resolution_testing)
        G_ema.eval()
        G_ema.requires_grad_(False)

        if rank == 0:
            print('Loading saved checkpoints...')
        # resume_niter = load_checkpoint(
        _ = load_checkpoint(
            module,
            G_ema,
            checkpoint_path=args.evaluate,
            optimizer=opt,
            rank=rank,
        )
        if rank == 0:
            print('Loaded saved checkpoints from {}!'.format(args.evaluate))

        print('Evaluating')
        args.run_dir = os.path.join(args.run_dir, 'testing_videos{}_views'.format(valid_pose_num))
        os.makedirs(args.run_dir, exist_ok=True)
        print(args.run_dir)
        print('Loading testing prompt set...')
        test_prompt_set = TextsPrompts(start=0.0, end=1.0)
        valid_prompt_loader = torch.utils.data.DataLoader(dataset=test_prompt_set, sampler=None, shuffle=False,
                                                          batch_size=1,
                                                          num_workers=args.num_workers, pin_memory=pin_memory,
                                                          prefetch_factor=2, worker_init_fn=_init_fn)
        G_val = G_ema
        G_val.requires_grad_(False)
        G_val.train(False)
        valid_prompts = []
        # torch.cuda.empty_cache()
        all_pose_imgs = []
        all_pose_depths = []
        all_pose_norms = []
        prompt_to_triplane = {}
        order_testing = 0
        for valid_prompts_imgs in valid_prompt_loader:
            valid_prompts.extend(valid_prompts_imgs)
            valid_prompts_planes = valid_prompts_imgs

            print('Starting to extract text embedding:', time.time())
            prompt_embeddings_planes = loss.guidance.get_text_embeds(valid_prompts_planes)
            text_tokens_planes = clip.tokenize(valid_prompts_planes).to(device)
            text_embeds_planes_eot = loss.clip_model.encode_text(text_tokens_planes).detach()
            # torch.cuda.empty_cache()

            pose_imgs, pose_depths, pose_norms, triplanes = G_val.valid_rendering(valid_prompts_planes, prompt_embeddings_planes, text_embeds_planes_eot, valid_pose_loader, random_bg=args.random_bg, max_depth=args.max_depth, cur_niter=None)

            all_pose_imgs.append(pose_imgs)
            all_pose_depths.append(pose_depths)
            all_pose_norms.append(pose_norms)

            images = torch.cat(all_pose_imgs, dim=0).cpu().numpy()
            images_depth = - torch.cat(all_pose_depths, dim=0).cpu().numpy()
            images_norm = torch.cat(all_pose_norms, dim=0).cpu().numpy()
            grid_size = (valid_pose_num, images.shape[0])
            print('Start to save fake image/depth...')
            save_video_grid(images.reshape(-1, 3, args.resolution_testing, args.resolution_testing), os.path.join(args.run_dir, f'fakes{order_testing:02d}.png'), drange=[-1, 1], grid_size=grid_size)
            save_video_grid(images_depth.reshape(-1, 1, args.resolution_testing, args.resolution_testing), os.path.join(args.run_dir, f'fakes{order_testing:02d}_depth.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)
            save_video_grid(images_norm.reshape(-1, 3, args.resolution_testing, args.resolution_testing), os.path.join(args.run_dir, f'fakes{order_testing:02d}_norm.png'), drange=[images_norm.min(), images_norm.max()], grid_size=grid_size)
            print('Image/depth saved!')

            del images, images_depth, images_norm
            all_pose_imgs = []
            all_pose_depths = []
            all_pose_norms = []
            order_testing += 1
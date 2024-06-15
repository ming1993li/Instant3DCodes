# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import cv2
from PIL import Image

import torch
import clip, os
from models.sd.sd import StableDiffusion

try:
    from apex import amp
except ImportError:
    pass
from torch import distributed as dist

import PIL.Image


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps, max=1e32))


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def interpolate_text_embeddings_over_views(all_embeddings, all_azimuths):
    left = 90
    right = 270
    view = 90
    front = 180

    interpolated_embeddings = []
    for b in range(all_azimuths.shape[0]):
        azimuth_val = all_azimuths[b]
        embeddings = {key: value[b].unsqueeze(0) for key, value in all_embeddings.items()}
        if azimuth_val >= left and azimuth_val < right:
            if azimuth_val >= front:
                r = 1 - (azimuth_val - front) / view
            else:
                r = 1 + (azimuth_val - front) / view
            start_z = embeddings['front']
            end_z = embeddings['side']
        else:
            if azimuth_val >= front:
                r = 1 - (azimuth_val - right) / view
            else:
                r = 1 + (azimuth_val - left) / view
            start_z = embeddings['side']
            end_z = embeddings['back']

        interpolated_embeddings.append(r * start_z + (1 - r) * end_z)
    return torch.cat(interpolated_embeddings, dim=0)


def interpolate_text_embeddings_across_prompts(prompt_embeddings, cur_niter, Dirichlet_alpha_range, Dirichlet_alpha_duration):
    B = prompt_embeddings.shape[0]
    rand_indices = torch.randperm(B)
    prompt_embeddings_reordered = prompt_embeddings[rand_indices, :]
    if cur_niter > Dirichlet_alpha_duration:
        alpha = Dirichlet_alpha_range[1]
    else:
        alpha = Dirichlet_alpha_range[0] + (Dirichlet_alpha_range[1] - Dirichlet_alpha_range[0]) / Dirichlet_alpha_duration * cur_niter
    coefficents = torch.tensor(np.random.dirichlet(alpha=[alpha, alpha], size=B), device=prompt_embeddings.device, dtype=prompt_embeddings.dtype)
    prompt_embeddings = prompt_embeddings * coefficents[:, 0][:, None, None] + prompt_embeddings_reordered * coefficents[:, 1][:, None, None]
    return prompt_embeddings, rand_indices, coefficents[:, 0].cpu().numpy()


def Bern_sample_prompts(prompts, rand_indices, coefficents):
    results = []
    for i in range(len(prompts)):
        if np.random.binomial(1, coefficents[i], 1)[0] == 1:
            results.append(prompts[i])
        else:
            results.append(prompts[rand_indices[i]])
    return results


class RenderingLoss:
    def __init__(self, device, rank, amp_train, amp_level, guid_name='sd', neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None,
                 neural_rendering_resolution_fade_kiter=0, pretrained_path=None, t_range=[0.02, 0.98]):
        super().__init__()
        self.device = device
        self.rank = rank
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kiter = neural_rendering_resolution_fade_kiter

        # clip model
        # clip_model, _ = clip.load("ViT-B/32", device=device)
        clip_model, _ = clip.load("ViT-L/14", device=device, download_root=pretrained_path)
        clip_model.requires_grad_(False)
        clip_model.eval()
        self.clip_model = clip_model

        # Construct stable diffusion.
        self.guid_name = guid_name
        guidance = StableDiffusion(device, '2.1', None, rank, download_path=pretrained_path, t_range=t_range)

        guidance.requires_grad_(False)
        guidance.eval()
            
        if amp_train:
            self.guidance = amp.initialize(guidance, opt_level=amp_level)
        else:
            self.guidance = guidance
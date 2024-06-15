# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import numpy as np
import time
import torch.nn.functional as F
import torch, random
import torch.nn as nn
from training.volumetric_rendering.renderer import ImportanceRenderer
from models.triplane_decoder import TriplaneDecoder
from diffusers.utils.import_utils import is_xformers_available

from training.generator_modules import EoTTransfer, Emb2EotTransfer, safe_normalize, FullyConnectedLayer, Transformer


class TriPlaneGenerator(torch.nn.Module):
    def __init__(self, rendering_kwargs={}, density_activation='softplus', guidance='sd'):
        super().__init__()
        divde = 4
        layers_per_block = 12
        attention_head_dim = [5, 10, 20, 20]
        norm_num_groups = int(32 / divde)

        plane_ch = 32
        decoder_output_dim = 3
        eot_channel = 512
        text_dim = 1024

        triplane_net = TriplaneDecoder(sample_size=256, in_channels=eot_channel, out_channels=(3 * plane_ch),
                                    attention_head_dim=attention_head_dim,
                                    block_out_channels=[int(320 / divde), int(640 / divde), int(1280 / divde),
                                                        int(1280 / divde)], norm_num_groups=norm_num_groups,
                                    cross_attention_dim=text_dim, use_linear_projection=True,
                                    layers_per_block=layers_per_block)
        # assert is_xformers_available()
        # triplane_net.set_use_memory_efficient_attention_xformers(True)
        self.triplane_net = triplane_net
        self.renderer = ImportanceRenderer()

        self.text_eot_transfer = EoTTransfer(in_dim=1024, dim=192, depth=1, eot_channel=eot_channel)
        self.text_emb2eot = Emb2EotTransfer(in_dim=1024, dim=192, depth=1, out_channel=768)

        self.decoder = OSGDecoder(plane_ch, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                                             'decoder_output_dim': decoder_output_dim},
                                  density_activation=density_activation, guidance=guidance)

        self.neural_rendering_resolution = 96
        self.rendering_kwargs = rendering_kwargs

    def update_resolution(self, resolution):
        self.neural_rendering_resolution = resolution
        self.rendering_kwargs['image_resolution'] = resolution

    def forward(self, prompts_planes, imgs, text_embeds, text_embeds_eot, ray_origins, ray_directions, sample=False,
                training_identity=False, random_bg=False, light_d=None, ambient_ratio=1.0, shading='albedo',
                max_depth=10.0, depth_scale=None, cur_niter=None):

        text_embeds_eot = self.text_emb2eot(text_embeds)
        text_embeds_eot_noise = torch.cat([text_embeds_eot, torch.randn_like(text_embeds_eot)], dim=1)

        if sample:
            return self.sample(prompts_planes, imgs, text_embeds, text_embeds_eot, text_embeds_eot_noise, ray_origins, ray_directions, cur_niter=cur_niter)

        planes = self.triplane_net(self.text_eot_transfer(text_embeds), text_embeds, text_embeds_eot_noise).sample

        assert (not sample)
        B, N, _ = ray_origins.shape

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        feature_samples, depth_samples, weights_samples, _, loss_orient, loss_smooth = self.renderer(planes, self.decoder, ray_origins, ray_directions,
                          self.rendering_kwargs, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading, cur_niter=cur_niter)  # channels last

        if random_bg == 'image':
            if random.random() < 1.5:
                bg_color = torch.rand((B, 1, feature_samples.shape[-1]), dtype=feature_samples.dtype).to(
                    feature_samples.device).repeat((1, N, 1))
                bg_color = bg_color * 2 - 1
                feature_samples = feature_samples + (1 - weights_samples) * bg_color
                bg_color = ((bg_color + 1) / 2)[:, 0, :3].unsqueeze(-1).unsqueeze(-1)
            else:
                bg_color = 0
            
        elif random_bg == 'pixel':
            bg_color = torch.rand_like(feature_samples)
            bg_color = bg_color * 2 - 1
            feature_samples = feature_samples + (1 - weights_samples) * bg_color
            bg_color = ((bg_color + 1) / 2)[:, :, :3].reshape(
                (B, self.neural_rendering_resolution, self.neural_rendering_resolution, 3)).permute(0, 3, 1, 2)
            bg_color = F.interpolate(bg_color, (256, 256), mode='bilinear', align_corners=False)
            
        else: # white bg
            # bg_color = torch.ones_like(feature_samples)
            # bg_color = bg_color * 2 - 1
            # feature_samples = feature_samples + (1 - weights_samples) * bg_color
            
            bg_color = None

        bg_depth = max_depth
        depth_samples = depth_samples + (1 - weights_samples) * bg_depth
        depth_samples = depth_samples * depth_scale.unsqueeze(-1)

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(B, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(B, 1, H, W)

        rgb_image = feature_image[:, :3]
        return {'rgb': rgb_image, 'depth': depth_image, 'weights_sum': weights_samples, 'bg_color': bg_color,
                'loss_orient': loss_orient, 'loss_smooth': loss_smooth}

    def sample(self, prompts_planes, imgs, text_embeds, text_embeds_eot, text_embeds_eot_noise, coordinates, directions, cur_niter=None):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes.
        planes = self.triplane_net(self.text_eot_transfer(text_embeds), text_embeds, text_embeds_eot_noise).sample
        planes = self.cross_att_triplane_texts(planes, text_embeds)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs, random_sample=True, cur_niter=cur_niter)

    def valid_rendering(self, prompts_planes, text_embeds, text_embeds_eot, valid_pose_loader, random_bg=False, max_depth=10.0, cur_niter=None):
        B = len(prompts_planes)
        text_embeds_eot = self.text_emb2eot(text_embeds)
        text_embeds_eot_noise = torch.cat([text_embeds_eot, torch.randn_like(text_embeds_eot)], dim=1)
        # text_embeds_eot_noise = torch.cat([text_embeds_eot, torch.zeros_like(text_embeds_eot)], dim=1)

        planes = self.triplane_net(self.text_eot_transfer(text_embeds), text_embeds, text_embeds_eot_noise).sample
        # print('Successfully generating Tri-plane:', time.time())

        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        # torch.cuda.empty_cache()
        all_pose_imgs = []
        all_pose_norms = []
        all_pose_depths = []
        for ray_o_d in valid_pose_loader:
            ray_o = ray_o_d['rays_o']
            ray_d = ray_o_d['rays_d']
            depth_scale = ray_o_d['depth_scale']
            ray_origins = ray_o.repeat((B, 1, 1)).to(planes.device)
            ray_directions = ray_d.repeat((B, 1, 1)).to(planes.device)
            depth_scale = depth_scale.repeat((B, 1)).to(planes.device)
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = (ray_origins[:, 0, :] + torch.randn_like(ray_origins[:, 0, :]))
            light_d = safe_normalize(light_d).unsqueeze(-1)

            feature_samples, depth_samples, weights_samples, normal_samples, _, _ = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs,
                      light_d=light_d, ambient_ratio=1.0, shading='albedo', cur_niter=cur_niter)  # channels last
            # print('Successfully rendering one view images:', time.time())
            if random_bg != 'none':
                # white bg for testing
                bg_color = torch.ones_like(feature_samples)
                bg_color = bg_color * 2 - 1
                feature_samples = feature_samples + (1 - weights_samples) * bg_color

                bg_color = torch.ones_like(normal_samples)
                bg_color = bg_color * 2 - 1
                normal_samples = normal_samples + (1 - weights_samples) * bg_color

            bg_depth = max_depth
            depth_samples = depth_samples + (1 - weights_samples) * bg_depth
            depth_samples = depth_samples * depth_scale.unsqueeze(-1)

            # torch.cuda.empty_cache()
            H = W = self.neural_rendering_resolution
            feature_image = feature_samples.permute(0, 2, 1).reshape(B, feature_samples.shape[-1], H, W).contiguous()

            normal_image = normal_samples.permute(0, 2, 1).reshape(B, normal_samples.shape[-1], H, W).contiguous()
            depth_image = depth_samples.permute(0, 2, 1).reshape(B, 1, H, W)

            rgb_image = feature_image[:, :3]
            all_pose_imgs.append(rgb_image)
            all_pose_norms.append(normal_image)
            all_pose_depths.append(depth_image)
        return torch.stack(all_pose_imgs, dim=1), torch.stack(all_pose_depths, dim=1), torch.stack(all_pose_norms, dim=1), planes.view(len(planes), -1, planes.shape[-2], planes.shape[-1])


class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options, density_activation, guidance):
        super().__init__()
        self.hidden_dim = 64
        self.density_activation = density_activation
        self.guidance = guidance
        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'],
                                lr_multiplier=options['decoder_lr_mul'])
        )

    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N * M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        
        if not self.training:
            rgb = torch.sigmoid(x[..., 1:]) * (1 + 2 * 0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
        else:
            rgb = x[..., 1:] * 0.5 + 0.5

        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}

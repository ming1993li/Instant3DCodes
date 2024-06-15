# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""

import torch

from training.volumetric_rendering.ray_marcher import MipRayMarcher2
from training.volumetric_rendering import math_utils
import torch.nn.functional as F

from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd


class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(max=15))


trunc_exp = _trunc_exp.apply


def biased_softplus(x, bias=0):
    return torch.nn.functional.softplus(x - bias)


def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                             [0, 1, 0],
                             [1, 0, 0]]], dtype=torch.float32)


def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    try:
        inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    except:
        inv_planes = torch.inverse(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    aabb = torch.tensor([-box_warp/2, -box_warp/2, -box_warp/2, box_warp/2, box_warp/2, box_warp/2], device=coordinates.device)
    coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds
    # coordinates = torch.min(torch.max(coordinates, aabb[:3]), aabb[3:]) # a manual clip.

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    with torch.cuda.amp.autocast(enabled=False):  # 禁止amp自动切换精度
        plane_features = plane_features.to(torch.float32)
        projected_coordinates = projected_coordinates.to(torch.float32)
        output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates, mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works df_if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    with torch.cuda.amp.autocast(enabled=False):  # 禁止amp自动切换精度
        grid = grid.to(torch.float32)
        coordinates = coordinates.to(torch.float32)
        sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=False)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features


class ImportanceRenderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = generate_planes()

    def forward(self, planes, decoder, ray_origins, ray_directions, rendering_options, light_d=None, ambient_ratio=1.0, shading='albedo', cur_niter=None):
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

        out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options, cur_niter=cur_niter)
        densities_coarse = out['sigma']
        # normals_coarse = self.finite_difference_normal(planes, decoder, sample_coordinates, sample_directions, rendering_options, cur_niter=cur_niter)
        normals_coarse = None

        if shading == 'albedo':
            colors_coarse = out['rgb']
        else:
            # lambertian shading
            if normals_coarse.shape[1] < 1e6:
                lambertian = (ambient_ratio + (1 - ambient_ratio) * torch.bmm(normals_coarse, light_d).clamp(min=0.1)).squeeze(-1)  # [N,]
                if shading == 'textureless':
                    colors_coarse = lambertian.unsqueeze(-1).repeat(1, 1, 3)
                elif shading == 'normal': # notice
                    colors_coarse = (normals_coarse + 1) / 2
                else:  # 'lambertian'
                    colors_coarse = out['rgb'] * lambertian.unsqueeze(-1)
            else:
                colors_coarse = out['rgb']

        # # orientation loss (not very exact in cuda ray mode)
        # loss_orient_coarse = (1 - torch.exp(- densities_coarse.squeeze(-1))).detach() * (normals_coarse * sample_directions).sum(-1).clamp(min=0) ** 2
        loss_orient_coarse = 0

        # # surface normal smoothness
        # normals_perturb = self.finite_difference_normal(planes, decoder, sample_coordinates + torch.randn_like(sample_coordinates) * 1e-2, sample_directions, rendering_options, cur_niter=cur_niter)
        # loss_smooth_coarse = (normals_coarse - normals_perturb).abs()
        loss_smooth_coarse = 0

        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        # normals_coarse = normals_coarse.reshape(batch_size, num_rays, samples_per_ray, normals_coarse.shape[-1])

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        _, _, weights, _ = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, normals_coarse, rendering_options)

        depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

        out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options, cur_niter=cur_niter)
        densities_fine = out['sigma']
        # normals_fine = self.finite_difference_normal(planes, decoder, sample_coordinates, sample_directions, rendering_options, cur_niter=cur_niter)
        normals_fine = None

        if shading == 'albedo':
            colors_fine = out['rgb']
        else:
            # lambertian shading
            if normals_fine.shape[1] < 1e6:
                lambertian = (ambient_ratio + (1 - ambient_ratio) * torch.bmm(normals_fine, light_d).clamp(min=0.1)).squeeze(-1)  # [N,]
                if shading == 'textureless':
                    colors_fine = lambertian.unsqueeze(-1).repeat(1, 1, 3)
                elif shading == 'normal':  # notice
                    colors_fine = (normals_fine + 1) / 2
                else:  # 'lambertian'
                    colors_fine = out['rgb'] * lambertian.unsqueeze(-1)
            else:
                colors_fine = out['rgb']

        # # orientation loss (not very exact in cuda ray mode)
        # loss_orient_fine = (1 - torch.exp(- densities_fine.squeeze(-1))).detach() * (normals_fine * sample_directions).sum(-1).clamp(min=0) ** 2
        loss_orient_fine = 0

        # # surface normal smoothness
        # normals_perturb = self.finite_difference_normal(planes, decoder, sample_coordinates + torch.randn_like(sample_coordinates) * 1e-2, sample_directions, rendering_options, cur_niter=cur_niter)
        # loss_smooth_fine = (normals_fine - normals_perturb).abs()
        loss_smooth_fine = 0.0

        colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
        densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)
        # normals_fine = normals_fine.reshape(batch_size, num_rays, N_importance, normals_fine.shape[-1])

        # all_depths, all_colors, all_densities, all_normals = self.unify_samples(depths_coarse, colors_coarse, densities_coarse, normals_coarse,
        #                                                       depths_fine, colors_fine, densities_fine, normals_fine)

        all_depths, all_colors, all_densities, all_normals = self.unify_samples(depths_coarse, colors_coarse, densities_coarse, depths_coarse,
                                                              depths_fine, colors_fine, densities_fine, depths_fine)

        # Aggregate
        rgb_final, depth_final, weights, normal_final = self.ray_marcher(all_colors, all_densities, all_depths, all_normals, rendering_options)

        # loss_orient = torch.cat([loss_orient_coarse, loss_orient_fine], dim=1).mean()
        # loss_smooth = torch.cat([loss_smooth_coarse, loss_smooth_fine], dim=1).mean()

        # return rgb_final, depth_final, weights.sum(2), normal_final, loss_orient, loss_smooth
        return rgb_final, depth_final, weights.sum(2), normal_final, 0, 0

    def run_model(self, planes, decoder, sample_coordinates, sample_directions, options, random_sample=False, cur_niter=None):
        sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'])

        out = decoder(sampled_features, sample_directions, cur_niter=cur_niter)
        density_activation = decoder.density_activation
        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']

        if not random_sample:
            out['sigma'] = out['sigma'] + self.density_blob(sample_coordinates, options['blob_density'], options['blob_radius'], density_activation).unsqueeze(-1)

        return out

    @torch.no_grad()
    def density_blob(self, x, blob_density, blob_radius, density_activation='softplus'):
        # x: [B, N, 3]

        d = (x ** 2).sum(-1)

        if density_activation == 'exp':
            g = blob_density * torch.exp(- d / (2 * blob_radius ** 2))
        else:
            g = blob_density * (1 - torch.sqrt(d) / blob_radius)
        return g

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, normals1, depths2, colors2, densities2, normals2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)
        all_normals = None
        if normals1 is not None and normals2 is not None:
            all_normals = torch.cat([normals1, normals2], dim=-2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        if normals1 is not None and normals2 is not None:
            all_normals = torch.gather(all_normals, -2, indices.expand(-1, -1, -1, all_normals.shape[-1]))

        return all_depths, all_colors, all_densities, all_normals

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1, 2, 0, 3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples

    def finite_difference_normal(self, planes, decoder, sample_coordinates, sample_directions, rendering_options, epsilon=1e-2, cur_niter=None):
        # sample_coordinates: [B, N, 3]
        x = sample_coordinates
        dx_pos = self.run_model(planes, decoder,
            (x + torch.tensor([[[epsilon, 0.00, 0.00]]], device=x.device)), sample_directions, rendering_options, cur_niter=cur_niter)['sigma']
        dx_neg = self.run_model(planes, decoder,
            (x + torch.tensor([[[-epsilon, 0.00, 0.00]]], device=x.device)), sample_directions, rendering_options, cur_niter=cur_niter)['sigma']
        dy_pos = self.run_model(planes, decoder,
            (x + torch.tensor([[[0.00, epsilon, 0.00]]], device=x.device)), sample_directions, rendering_options, cur_niter=cur_niter)['sigma']
        dy_neg = self.run_model(planes, decoder,
            (x + torch.tensor([[[0.00, -epsilon, 0.00]]], device=x.device)), sample_directions, rendering_options, cur_niter=cur_niter)['sigma']
        dz_pos = self.run_model(planes, decoder,
            (x + torch.tensor([[[0.00, 0.00, epsilon]]], device=x.device)), sample_directions, rendering_options, cur_niter=cur_niter)['sigma']
        dz_neg = self.run_model(planes, decoder,
            (x + torch.tensor([[[0.00, 0.00, -epsilon]]], device=x.device)), sample_directions, rendering_options, cur_niter=cur_niter)['sigma']

        normal = torch.concat([
            0.5 * (dx_pos - dx_neg) / epsilon,
            0.5 * (dy_pos - dy_neg) / epsilon,
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        normal = safe_normalize(-normal)
        normal = torch.nan_to_num(normal)

        return normal
    # def finite_difference_normal(self, planes, decoder, sample_coordinates, sample_directions, rendering_options, epsilon=1e-2, cur_niter=None):
    #     x = sample_coordinates
    #     with torch.enable_grad():
    #         with torch.cuda.amp.autocast(enabled=False):
    #             x.requires_grad_(True)
    #             sigma = self.run_model(planes, decoder, x, sample_directions, rendering_options, cur_niter=cur_niter)['sigma']
    #             # query gradient
    #             normal = - torch.autograd.grad(torch.sum(sigma), x, create_graph=True)[0]  # [N, 3]
    #
    #     normal = safe_normalize(normal)
    #     normal = torch.nan_to_num(normal)
    #     return normal


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))
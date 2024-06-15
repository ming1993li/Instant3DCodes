import os
import cv2
import glob
import json
import tqdm
import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
import trimesh

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from packaging import version as pver


DIR_COLORS = np.array([
    [255, 0, 0, 255], # front
    [0, 255, 0, 255], # side
    [0, 0, 255, 255], # back
    [255, 255, 0, 255], # side
    [255, 0, 255, 255], # overhead
    [0, 255, 255, 255], # bottom
], dtype=np.uint8)


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


def linear_enlarge_ranges(curr, duration, end_range, start_range):
    return [np.linspace(start_range[0], end_range[0], int(duration * 1000))[curr], np.linspace(start_range[1], end_range[1], int(duration * 1000))[curr]]


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        if error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = - torch.ones_like(i)
    xs = - (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    scale = 1 / directions.pow(2).sum(-1).pow(0.5)

    # directions = safe_normalize(directions)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)
    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    results['depth_scale'] = scale

    return results


def visualize_poses(poses, dirs, size=0.1):
    # poses: [B, 4, 4], dirs: [B]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose, dir in zip(poses, dirs):
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)

        # different color for different dirs
        segs.colors = DIR_COLORS[[dir]].repeat(len(segs.entities), 0)

        objects.append(segs)

    trimesh.Scene(objects).show()

# def get_view_direction(thetas, phis, overhead, front): # origin
#     #                   phis [B,] 0-360         thetas: [B,] 0-120
#     # front = 0         [0, front)
#     # side (left) = 1   [front, 180)
#     # back = 2          [180, 180+front)
#     # side (right) = 3  [180+front, 360)
#     # top = 4                               [0, overhead]
#     # bottom = 5                            [180-overhead, 180]
#     res = torch.zeros(thetas.shape[0], dtype=torch.long)
#     # first determine by phis
#     res[(phis < front)] = 0
#     res[(phis >= front) & (phis < np.pi)] = 1
#     res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
#     res[(phis >= (np.pi + front))] = 3
#     # override by thetas
#     res[thetas <= overhead] = 4
#     res[thetas >= (np.pi - overhead)] = 5
#     return res


def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,] 0-360         thetas: [B,] 0-120
    # front = 0         [180-front/2, 180+front/2)
    # side (left) = 1   [180+front/2, 360-front/2)
    # back = 2          [360-front/2, 360) or [0, front/2)
    # side (right) = 3  [front/2, 180-front/2)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    half = front / 2
    res[((phis >= (np.pi - half)) & (phis < (np.pi + half)))] = 0
    res[((phis >= (np.pi + half)) & (phis < (np.pi * 2 - half)))] = 1
    res[((phis < half) | (phis >= (np.pi * 2 - half)))] = 2
    res[((phis >= half) & (phis < (np.pi - half)))] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def rand_poses(size, device, radius_range=[1, 1.5], theta_range=[0, 120], phi_range=[0, 360], return_dirs=False, angle_overhead=30, angle_front=60, jitter=False, uniform_sphere_rate=0.5):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)
    
    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                (torch.rand(size, device=device) - 0.5) * 2.0,
                torch.rand(size, device=device),
                (torch.rand(size, device=device) - 0.5) * 2.0,
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:,1])
        phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]

        # phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
        if phi_range[1] <= np.deg2rad(240.0) and phi_range[0] >= np.deg2rad(120.0):
            phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
        else:
            rand = random.random()  # notice
            if rand > 1.0 - 0.2 / 2: # 0.2 back 90度
                phis = torch.rand(size, device=device) * (phi_range[1] - np.deg2rad(315.0)) + np.deg2rad(315.0)
            elif rand > 1.0 - 0.2: # 0.2 back 90度
                phis = torch.rand(size, device=device) * (np.deg2rad(45.0) - phi_range[0]) + phi_range[0]
            elif rand > 0.8 - 0.2: # 0.2 right 90度
                phis = torch.rand(size, device=device) * (np.deg2rad(315.0) - np.deg2rad(225.0)) + np.deg2rad(225.0)
            elif rand > 0.6 - 0.2: # 0.2 left 90度
                phis = torch.rand(size, device=device) * (np.deg2rad(135.0) - np.deg2rad(45.0)) + np.deg2rad(45.0)
            else:            # 0.4 front 90度
                phis = torch.rand(size, device=device) * (np.deg2rad(225.0) - np.deg2rad(135.0)) + np.deg2rad(135.0)

            # phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1) # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    
    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None
    
    return poses, dirs, thetas, phis, radius


def circle_poses(device, radius=1.25, theta=60, phi=0, return_dirs=False, angle_overhead=30, angle_front=60):

    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None
    
    return poses, dirs    


def sphere_cosine_distance(refer_phi, refer_theta, phis, thetas):
    lam1 = refer_phi
    lam2 = phis
    pusei1 = torch.deg2rad(torch.tensor(90)) - refer_theta
    pusei2 = torch.deg2rad(torch.tensor(90)) - thetas
    cosine = torch.sin(pusei1) * torch.sin(pusei2) + torch.cos(pusei1) * torch.cos(pusei2) * torch.cos(lam2 - lam1)
    return cosine


class NeRFDataset:
    def __init__(self, opt, device='cpu', type='train', H=256, W=256, size=100):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.H = H
        self.W = W
        self.size = size

        self.training = self.type in ['train', 'all']

        self.cx = self.H / 2
        self.cy = self.W / 2
        self.cur_niter = 0

        # # [debug] visualize poses
        # poses, dirs = rand_poses(100, self.device, theta_range=[90, 90], phi_range=[0, 0], uniform_sphere_rate=0, radius_range=self.opt.radius_range, return_dirs=True, jitter=self.opt.jitter_pose)
        # visualize_poses(poses.detach().cpu().numpy(), dirs.detach().cpu().numpy())

    def __len__(self):
        return self.size

    def collate(self, index):
        if self.training:
            pass
        else:
            # print(index)
            assert len(index) == 1  # always 1
            # # circle pose
            # if index[0] == 0:
            #     phi = (self.opt.refer_phi_range[0] + self.opt.refer_phi_range[1]) / 2
            #     radius = (self.opt.refer_radius_range[0] + self.opt.refer_radius_range[1]) / 2
            # else:
            #     phi = ((index[0] - 1) / (self.size - 1)) * 360
            phi = ((index[0] - 0) / (self.size - 0)) * 360 + 180

            radius = (self.opt.refer_radius_range[0] + self.opt.refer_radius_range[1]) / 2 * 1.3
            # theta = 60
            theta = (self.opt.refer_theta_range[0] + self.opt.refer_theta_range[1]) / 2

            poses, dirs = circle_poses(self.device, radius=radius, theta=theta, phi=phi, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)

            # fixed focal
            fov = (self.opt.fovy_range[1] + self.opt.fovy_range[0]) / 2

        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])

        # sample a low-resolution but full image
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)
        rays['dirs'] = dirs
        # rays['rays_o'] [B, 4096, 3]
        # rays['rays_d'] [B, 4096, 3]
        return rays

    def dataloader(self, sampler=None, shuffle=False, batch_size=1, drop_last=False, worker_init_fn=None, pin_memory=False, num_workers=8):
        if not self.training:
            assert batch_size == 1
        if self.training:
            # notice: num_workers must be zero when training for pose of all views
            num_workers = 0
        loader = DataLoader(list(range(self.size)), batch_size=batch_size, sampler=sampler, collate_fn=self.collate, shuffle=shuffle, num_workers=num_workers,
                            drop_last=drop_last, worker_init_fn=worker_init_fn, pin_memory=pin_memory)
        return loader

import glob
import os, math
from args import args
os.environ['NCCL_P2P_DISABLE'] = 'NVL'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import torch
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
import torch.distributed as dist

import sys
sys.path.append(os.path.dirname(sys.path[0]))
import json, datetime
import tempfile
import torch
import warnings, shutil
warnings.filterwarnings('ignore')
import dnnlib
from training import training_engine
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops


def launch_training(args, desc, outdir):
    dnnlib.util.Logger(should_flush=True)
    curr_path = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(curr_path, outdir)
    args.run_dir = os.path.join(outdir, f'{str(datetime.datetime.now())[:19]}-{desc}').replace(':', '-')

    # Print options.
    if args.local_rank == 0:
        print()
        print('Training options:')
        # print(json.dumps(args.__dict__, indent=2))
        print()
        print(f'Output directory:    {args.run_dir}')
        print(f'Number of GPUs:      {args.num_gpus}')
        print(f'Batch size:          {args.batch_size}')
        print()

    # Create output directory.
    if args.local_rank == 0:
        print('Creating output directory...')
        os.makedirs(args.run_dir, exist_ok=True)
        with open(os.path.join(args.run_dir, 'args.json'), 'wt') as f:
            json.dump(args.__dict__, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Execute training loop.
    training_engine.training_loop(rank=args.local_rank, args=args)

    
def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    
def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def init():
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'

    backend = 'gloo' if os.name == 'nt' else 'nccl'
    torch.distributed.init_process_group(backend=backend, init_method='env://', timeout=datetime.timedelta(seconds=7200000),)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

    sync_device = torch.device('cuda') if get_world_size() > 1 else None
    training_stats.init_multiprocessing(rank=get_rank(), sync_device=sync_device)


def main(args):
    # Init torch.distributed.
    args.num_gpus = len(args.gpu_ids.split(','))
    # # rank = args.local_rank
    # rank = get_rank()
    # # dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=5400), rank=rank, world_size=args.num_gpus)
    # dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=args.num_gpus)
    # # dist.init_process_group(backend='gloo', init_method='env://', rank=rank, world_size=args.num_gpus)
    # torch.cuda.set_device(args.local_rank)
    
    # torch.multiprocessing.set_start_method('spawn')
    init()
    args.local_rank = rank = get_rank()

    # Init torch_utils.
    sync_device = torch.device('cuda', rank)
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Loss
    args.loss_kwargs = dnnlib.EasyDict()
        
    # Hyperparameters & settings.
    args.batch_gpu = args.batch_size // args.num_gpus
    args.lr = args.lr * args.batch_size ** 0.5 # improve
    args.opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=args.lr, betas=(0.9, 0.99), eps=1e-15, weight_decay=0)  # LatentNeRF

    # Sanity checks.
    if args.batch_size % args.num_gpus != 0:
        raise ValueError('--batch must be a multiple of --gpus')
    if args.batch_size % (args.num_gpus * args.batch_gpu) != 0:
        raise ValueError('--batch must be a multiple of --gpus times --batch-gpu')
    if any(not metric_main.is_valid_metric(metric) for metric in args.metrics):
        raise ValueError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    args.ema_kiter = args.batch_size * 10 / 32
    args.ema_rampup = 0.05     # EMA ramp-up coefficient. None = no rampup.

    rendering_options = {
        'image_resolution': args.neural_rendering_resolution_initial,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',
        'density_reg': args.density_reg, # strength of density regularization
        'density_reg_p_dist': args.density_reg_p_dist, # distance at which to sample perturbed points for density regularization
        'reg_type': args.reg_type, # for experimenting with variations on density regularization
        'decoder_lr_mul': args.decoder_lr_mul, # learning rate multiplier for decoder
        'blob_density': args.blob_density,
        'blob_radius': args.blob_radius
    }

    rendering_options.update({
        'depth_resolution': 64,  # number of uniform samples to take per ray.
        'depth_resolution_importance': 64, # number of importance samples to take per ray.
        # 'ray_start': 0.1, # near point along each ray to start taking samples.
        'ray_start': 'auto', # near point along each ray to start taking samples.
        # 'ray_end': 2.6, # far point along each ray to stop taking samples.
        'ray_end': 'auto', # far point along each ray to stop taking samples.
        # 'box_warp': 1.6, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
        'box_warp': 3.5, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
        'white_back': False,
        # 'avg_camera_radius': 1.7, # used only in the visualizer to specify camera orbit radius.
        # 'avg_camera_pivot': [0, 0, 0],  # used only in the visualizer to control center of camera rotation.
    }) # improve

    if args.density_reg > 0:
        args.reg_interval = args.density_reg_every
    args.rendering_kwargs = rendering_options

    args.loss_kwargs.neural_rendering_resolution_initial = args.neural_rendering_resolution_initial
    args.loss_kwargs.neural_rendering_resolution_final = args.neural_rendering_resolution_final
    args.loss_kwargs.neural_rendering_resolution_fade_kiter = args.neural_rendering_resolution_fade_kiter

    if args.nobench:
        args.cudnn_benchmark = False
    # Description string.
    desc = 'testing'
    # Launch.
    launch_training(args=args, desc=desc, outdir=args.outdir)


if __name__ == "__main__":
    # Set seed
    # set_random_seed(args.random_seed)
    main(args) # pylint: disable=no-value-for-parameter

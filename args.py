import argparse
import math

parser = argparse.ArgumentParser()
# Generalpod
parser.add_argument('--desc', help='modification description', default='',)
parser.add_argument('--pretrained_path', help='pretrained weights for large models', default='/home/liming/projects/Text23D_Downloaded_weights',)
parser.add_argument('--outdir', help='Where to save the results', default='./experiments', type=str)
parser.add_argument('--gpu-ids', help='GPU IDs', type=str, default='1,2')
parser.add_argument('--batch-size', help='Total batch size', type=int)
parser.add_argument('--sd_steps', help='stable diffusion denoise steps', default=50, type=int)
parser.add_argument('--guidance', help='guidance manner', default='sd', type=str)
parser.add_argument('--guid_scale', help='sds loss guidance scale', default=100, type=float)
parser.add_argument('--random_bg', help='random background color, DO NOT SELECT PIXEL', default='image', type=str, choices=['pixel', 'image', 'none'])
parser.add_argument('--no_plane_prompt', help='do not use plane prompt', default=False, type=bool)
# Training
parser.add_argument('--amp_level', help='Mixed precision level: O1 (FP16 and FP32), O0 (FP32)', type=str, default='O1')
parser.add_argument('--amp_train', help='Whether use Mixed precision training', type=bool, default=False)
parser.add_argument('--lr', help='learning rate  [default: varies]', type=float, default=1e-5)
parser.add_argument('--metrics', help='Quality metrics', type=list, default=['fid50k_full'])
parser.add_argument('--pin_memory', help='True: fast with more gpu memory', type=bool, default=True)
parser.add_argument('--random_seed', help='Random seed', type=int, default=42)
parser.add_argument('--nobench', help='Disable cuDNN benchmarking', type=bool, default=False)
parser.add_argument('--cudnn_benchmark', help='Improves training speed', type=bool, default=True)
parser.add_argument('--num_workers', help='DataLoader worker number', type=int, default=8)

# Evaluating model weights
parser.add_argument('--evaluate', help='Saved model weights for evaluating', type=str, default='./model-pretrained.pth')

# Camera pose sampling
parser.add_argument('--max_depth', type=float, default=10.0, help="farthest depth")
parser.add_argument('--density_activation', type=str, default='softplus', help="density activation function")
parser.add_argument('--blob_density', type=float, default=10, help="max (center) density for the density blob")
parser.add_argument('--blob_radius', type=float, default=0.6, help="control the radius for the density blob")
parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")
parser.add_argument('--uniform_sphere_rate', type=float, default=0.0,
                    help="likelihood of sampling camera location uniformly on the sphere surface area")
parser.add_argument('--radius_range', type=float, nargs='*', default=[3.0, 3.6], help="training camera radius range") # improve
parser.add_argument('--fovy_range', type=float, nargs='*', default=[70, 80], help="training camera fovy range") # improve
parser.add_argument('--theta_range', type=float, nargs='*', default=[35, 125], help="training camera theta range") # improve
parser.add_argument('--phi_range', type=float, nargs='*', default=[0, 360], help="training camera theta range") # improve
parser.add_argument('--phi_range_warm', type=float, nargs='*', default=[135, 225], help="training camera phi range for warm") # improve
parser.add_argument('--angle_overhead', type=float, default=20, help="angle_overhead") # improve
parser.add_argument('--angle_front', type=float, default=90, help="angle_front") # improve
parser.add_argument('--duration_warm', type=int, default=0, help='decay kiter for sampling from reference view to all views')
parser.add_argument('--refer_radius_range', type=float, nargs='*', default=[3.3, 3.3], help="training camera radius range")
parser.add_argument('--refer_fovy_range', type=float, nargs='*', default=[75, 75], help="training camera fovy range")
parser.add_argument('--refer_theta_range', type=float, nargs='*', default=[90, 90], help="reference view 60 theta range")
parser.add_argument('--refer_phi_range', type=float, nargs='*', default=[180, 180], help="reference view 180 phi range")

# Rendering
parser.add_argument('--neural_rendering_resolution_initial', help='Resolution to render at', type=int, default=64)
parser.add_argument('--neural_rendering_resolution_final', help='Final resolution to render at, df_if blending', type=int, default=None)
parser.add_argument('--neural_rendering_resolution_fade_kiter', help='Kimg to blend resolution over', type=int, default=100)
parser.add_argument('--resolution_testing', help='Resolution to render for testing', type=int, default=128)
parser.add_argument('--decoder_lr_mul', help='decoder learning rate multiplier.', type=float, default=1)
parser.add_argument('--density_reg',    help='Density regularization strength.', type=float, default=0.25,)
parser.add_argument('--density_reg_p_dist',    help='density regularization strength.', type=float, default=0.004,)
parser.add_argument('--reg_type', help='Type of regularization', choices=['l1', 'l1-alt', 'monotonic-detach', 'monotonic-fixed', 'total-variation'], default='l1')
parser.add_argument('--density_reg_every',    help='frequency of density reg', type=int, default=4)

parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
args = parser.parse_args()

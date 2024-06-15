# Official Instant3D Codes
## This is the official implementation of our paper [Instant3D: Instant Text-to-3D Generation](https://arxiv.org/abs/2311.08403).

# Building the required Conda environment by requirements.txt

# Downloading the pre-trained ckpts
## Our Instant3D network [weights](https://drive.google.com/file/d/1SXQco_5n0yQBeuEsLO5-KR-jJQ8v9gxF/view?usp=sharing) 
(revise args.py --evaluate accordingly)
## Pretrained CLIP and Stable Diffusion weights (revise args.py ----pretrained_path)
### The CLIP weight for ViT-L/14
### downloading SD 2.1, sign in hugging face account through 
`from huggingface_hub import login
login('')`

# Install tiny-cuda-nn 

# DDP running
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2349 ./main.py --batch-size 1 --random_seed 666 --gpu-ids 0

## Citation
```
@article{li2023instant3d,
title     = {Instant3D: Instant Text-to-3D Generation},
author    = {Li, Ming and Zhou, Pan and Liu, Jia-Wei, and Keppo, Jussi and Lin, Min and Yan, Shuicheng and Xu, Xiangyu},
journal   = {International Journal of Computer Vision},
year      = {2024},
}
```

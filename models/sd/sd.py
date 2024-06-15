import os
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
from diffusers.models.unet_2d_condition import UNet2DConditionModel as UNetTriplans
from diffusers.utils.import_utils import is_xformers_available
logging.set_verbosity_error()

import torch, clip
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from torch.cuda.amp import custom_bwd, custom_fwd


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        # import pdb;
        # pdb.set_trace()
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.1', hf_key=None, rank=0, download_path=None, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        if rank == 0:
            print(f'[INFO] loading stable diffusion...')

        # # install git lfs
        # curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
        # sudo apt-get install git-lfs
        #
        # # manually download stable diffusion models
        # git lfs install
        # git clone https://huggingface.co/stabilityai/stable-diffusion-2-1-base
        # git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
        # git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
        # git clone https://huggingface.co/openai/clip-vit-large-patch14

        # # through shallow clone
        # git clone https://huggingface.co/stabilityai/stable-diffusion-2-1-base --depth 1
        # cd stable-diffusion-2-1-base
        # git fetch --unshallow
        from huggingface_hub import login
        login('')
        # download_path = None
        # if download_path is not None:
        #     if hf_key is not None:
        #         print(f'[INFO] using hugging face custom model key: {hf_key}')
        #         model_key = hf_key
        #     elif self.sd_version == '2.1':
        #         model_key = os.path.join(download_path, "stable-diffusion-2-1-base")
        #     elif self.sd_version == '2.0':
        #         model_key = os.path.join(download_path, "stable-diffusion-2-base")
        #     elif self.sd_version == '1.5':
        #         model_key = os.path.join(download_path, "stable-diffusion-v1-5")
        #     else:
        #         raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')
        # else:
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", cache_dir=download_path).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer", cache_dir=download_path)
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", cache_dir=download_path).to(self.device)
        unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", cache_dir=download_path) #block_out_channels (320, 640, 1280, 1280)
        assert is_xformers_available()
        # self.unet.enable_xformers_memory_efficient_attention()
        unet.set_use_memory_efficient_attention_xformers(True)
        self.unet = unet.to(self.device)

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", cache_dir=download_path)
        # self.scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        if rank == 0:
            print(f'[INFO] loaded stable diffusion!')
        self.aug = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def get_text_embeds(self, prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        return text_embeddings





import os, glob
import random
import torch
import PIL.Image as Image
import torchvision.transforms as T


class HumanPrompts(torch.utils.data.Dataset):
    curr_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(curr_path, 'prompts/portraits.txt')

    def __init__(self, start=0.0, end=0.8):
        with open(self.path, 'r') as f:
            prompts = f.readlines()
        prompts = [p[:-1] for p in prompts]
        self.prompts = prompts[int(len(prompts) * start): int(len(prompts) * end + 1)]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]
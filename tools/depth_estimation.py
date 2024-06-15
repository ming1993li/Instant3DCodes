from transformers import AutoImageProcessor, DPTForDepthEstimation
import torch, os
import numpy as np
from PIL import Image
import requests
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
pic = './yellow_dog.png'
image = Image.open(pic)

image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
# mean = std = [0.5, 0.5, 0.5] Resampling.BILINEAR
# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")
# inputs['pixel_values'] # shape [1, 3, 384, 384]
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# interpolate to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

# visualize the prediction
output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formatted)
depth.save('./yellow_dog_depth.png')
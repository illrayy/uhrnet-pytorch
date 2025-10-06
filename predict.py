import os
import cv2
import numpy as np
from tqdm import tqdm

import torch

from nets.uhrnet import UHRnet

backbone    = "UHRNet_W18_Small"
model_path  = "logs/last_model.pth"

img_path = 'F:/CropSeg/labeled/Grape/train/images'
output_path = 'F:/CropSeg/labeled/Grape/train/test_results'

num_classes = 2

input_shape = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UHRnet(num_classes=num_classes, backbone = backbone)
model.load_state_dict(torch.load(model_path, map_location = 'cpu'))
model.to(device)

os.makedirs(output_path, exist_ok=True)

vis_color = 255//num_classes

for img_name in tqdm(os.listdir(img_path)):
    img = cv2.imread(os.path.join(img_path, img_name))
    h, w = img.shape[:2]
    img = cv2.resize(img, (input_shape, input_shape))
    img = img.astype(np.float32)/255
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    output = model(img)
    output = output.argmax(dim=1).squeeze(0)
    output = output.cpu().numpy()
    output = cv2.resize(output.astype(np.uint8), (w, h))
    output = output * vis_color
    cv2.imwrite(os.path.join(output_path, img_name), output)
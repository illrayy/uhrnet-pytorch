import os
import random
import cv2
import numpy as np
import math
from functools import partial

import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from nets.uhrnet import UHRnet
from nets.uhrnet_training import CE_Loss, Focal_Loss

def get_lr_scheduler(lr, total_iters, warmup_total_iters, warmup_lr_start = 0):
    func = partial(warm_cos_lr, lr, total_iters, warmup_total_iters, warmup_lr_start,)
    return func

def warm_cos_lr(lr, total_iters, warmup_total_iters, warmup_lr_start, iters):
    """Cosine learning rate with warm up."""
    if iters <= warmup_total_iters:
        lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
    else:
        lr *= 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters)
            )
        )
    return lr

def set_optimizer_lr(optimizer, lr_scheduler_func, iter):
    lr = lr_scheduler_func(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class baseDataset(Dataset):
    def __init__(self, img_path, mask_path, input_shape, num_classes, ratio_range=(0.75, 1.5), cat_max_ratio=0.75, is_train=True):
        self.mask_path = []
        self.img_path = []
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.ratio_range = ratio_range
        self.is_train = is_train
        self.cat_max_ratio = cat_max_ratio

        mask_names = os.listdir(mask_path)
        for mask_name in mask_names:
            self.img_path.append(os.path.join(img_path, mask_name.replace('.png', '.jpg')))
            self.mask_path.append(os.path.join(mask_path, mask_name))

    def __len__(self):
        return len(self.img_path)
    
    def generate_crop_bbox(self, img):
        margin_h = max(img.shape[0] - self.input_shape, 0)
        margin_w = max(img.shape[1] - self.input_shape, 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.input_shape
        crop_x1, crop_x2 = offset_w, offset_w + self.input_shape
        return crop_y1, crop_y2, crop_x1, crop_x2
    
    def random_crop(self, img, mask):
        crop_bbox = self.generate_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = mask[crop_bbox[0]:crop_bbox[1], crop_bbox[2]:crop_bbox[3]]
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != 255]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.generate_crop_bbox(img)
        img = img[crop_bbox[0]:crop_bbox[1], crop_bbox[2]:crop_bbox[3]]
        mask = mask[crop_bbox[0]:crop_bbox[1], crop_bbox[2]:crop_bbox[3]]
        return img, mask
        
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.img_path[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path[idx])[:,:,0]
        mask[mask > 0] = 1

        if not self.is_train:
            img = img.astype(np.float32)/255
            return img, mask

        ratio = np.random.uniform(*self.ratio_range)

        img = cv2.resize(img, None, fx=ratio, fy=ratio)
        mask = cv2.resize(mask, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)

        img, mask = self.random_crop(img, mask)

        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        img = img.astype(np.float32)/255
        # pad to input_shape
        img = cv2.copyMakeBorder(img, 0, self.input_shape - img.shape[0], 0, self.input_shape - img.shape[1], cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, 0, self.input_shape - mask.shape[0], 0, self.input_shape - mask.shape[1], cv2.BORDER_CONSTANT, value=255)

        return img, mask

if __name__ == "__main__":

    #  Number of classes + background
    num_classes = 2

    #-------------------------------------------------------------------#
    #   UHRNet_W18_Small
    #   UHRNet_W48
    #-------------------------------------------------------------------#
    backbone    = "UHRNet_W18_Small"

    model_path  = "model_data/fcn_uhrnetw18_small_cityscapes_1024x512_120k.pth"

    # path of images. *.jpg
    img_path = 'F:/CropSeg/labeled/Grape/train/images'
    # path of masks. *.png
    mask_path = 'F:/CropSeg/labeled/Grape/train/masks'

    input_shape = 1024
    

    total_Epoch = 50
    warmup_epoch = 5
    batch_size = 2

    lr = 1e-4

    focal_loss = False


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = UHRnet(num_classes=num_classes, backbone = backbone)

    if model_path != '':
        print('Load weights {}.'.format(model_path))
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = 'cpu')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), 0, betas = (0.937, 0.999), weight_decay = 5e-4)
    
    train_dataset   = baseDataset(img_path=img_path, mask_path=mask_path, input_shape=input_shape, num_classes=num_classes, is_train=True)
    train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = 2, pin_memory=True,
                                drop_last = True)

    epoch_step = len(train_loader)//batch_size

    warmup_total_iters = epoch_step * warmup_epoch
    total_iters = epoch_step * total_Epoch
    lr_scheduler_func = get_lr_scheduler(lr=lr, warmup_total_iters=warmup_total_iters, total_iters=total_iters)

    iters_now = 0
    running_loss = 0
    for epoch in range(total_Epoch):  
        model.train()              
        for batch in train_loader:
            iters_now += 1
            imgs, masks = batch
            imgs, masks = imgs.to(device), masks.to(device).long()
            imgs = imgs.permute(0, 3, 1, 2)

            outputs = model(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, masks, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, masks, num_classes = num_classes)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            set_optimizer_lr(optimizer, lr_scheduler_func, iters_now)

            if iters_now % 100 == 0:
                running_loss = running_loss / 100
                print(f"Epoch: {epoch}, Iter: {iters_now}, Loss: {running_loss}")
                running_loss = 0
        torch.save(model.state_dict(), f"logs/last_model.pth")

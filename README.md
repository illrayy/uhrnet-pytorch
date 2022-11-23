# uhrnet-pytorch

## Reference
> Jian Wang, Xiang Long, Guowei Chen, Zewu Wu, Zeyu Chen, Errui Ding et al. "U-HRNet: Delving into Improving Semantic Representation of High Resolution Network for Dense Prediction" arXiv preprint arXiv:2210.07140 (2022).

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|FCN|UHRNet_W18_small|1024x512|80000|77.66%|78.26%|78.47%|
|FCN|UHRNet_W18_small|1024x512|120000|78.39%|79.09%|79.03%|
|FCN|UHRNet_W48|1024x512|80000|81.28%|81.76%|81.48%|
|FCN|UHRNet_W48|1024x512|120000|81.91%|82.39%|82.28%|

backbone部分的代码是基于paddleseg修改的

本仓库的架构基于[bubbliiiing](https://github.com/bubbliiiing) 的语义分割仓库，使用方法可参考bubbliiiing的[视频](https://space.bilibili.com/472467171)

数据集格式为VOC
Dataset format refers to VOC

#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from torchstat import stat

from nets.uhrnet import UHRnet

if __name__ == "__main__":
    model = UHRnet(num_classes=21, backbone='UHRNet_W48')
    stat(model, (3, 512, 512))
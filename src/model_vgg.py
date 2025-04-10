import torchvision.models as models
import torch.nn as nn
from torchvision.models import VGG16_BN_Weights

class VGG(nn.Module):
    def __init__(self, num_classes=9):  # FER+ 有 9 个情绪类别
        super(VGG, self).__init__()

        #下面这个写法是旧写法，这样写会报warning
        #self.base = models.vgg16_bn(pretrained=True)  # 加载 VGG16 带 BN

        #换成这种写法，加载 VGG16 带 BN
        self.base = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        self.base.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # 替换第一层适应灰度图

        # 替换 classifier
        self.base.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)  # 输出 soft labels
        )

    def forward(self, x):
        return self.base(x)

import torchvision.models as models
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, num_classes=8):  # FER+ 有 8 个情绪类别
        super(VGG, self).__init__()
        self.base = models.vgg16_bn(pretrained=True)  # 加载 VGG16 带 BN
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

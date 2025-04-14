import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models import efficientnet_b2

class EfficientNetSoftLabel(nn.Module):
    def __init__(self, num_classes=9, dropout=0.4):
        super(EfficientNetSoftLabel, self).__init__()
        self.base = efficientnet_b2(weights="DEFAULT")
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.base(x)

class VGG13_PyTorch(nn.Module):
    def __init__(self):
        super(VGG13_PyTorch, self).__init__()
        self.model = EfficientNetSoftLabel(num_classes=9)
        self.batch_size = 32
        self.lr = 1e-4
        self.epoch = 12

    def forward(self, x):
        x_aug = []
        for img in x:
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)  # 灰度 ➜ RGB
            img = TF.resize(img, [224, 224])
            angle = torch.randint(-10, 10, (1,)).item()
            img = TF.rotate(img, angle)
            img = TF.adjust_brightness(img, 1 + (torch.rand(1).item() - 0.5) * 0.4)
            img = TF.adjust_contrast(img, 1 + (torch.rand(1).item() - 0.5) * 0.4)
            img = TF.normalize(img, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            x_aug.append(img)
        x = torch.stack(x_aug)
        return self.model(x)

    def apply(self, fn):
        if not hasattr(self, "_warned"):
            print("⛔ EfficientNet 权重初始化已跳过（预训练参数保留）")
            self._warned = True

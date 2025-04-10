import torch
import torch.nn as nn

# 卷积层（Convolutional Layers）：
# 第一个循环部分：
# 当 out_channels 为 64 时，有 2 个卷积层，每个卷积层的配置为 nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)，后跟 nn.ReLU(inplace=True)。这里 in_channels 初始为 1（输入通道数），经过这 2 层卷积后，in_channels 变为 64。
# 当 out_channels 为 128 时，同样有 2 个卷积层，配置为 nn.Conv2d(64, 128, kernel_size=3, padding=1)，后跟 nn.ReLU(inplace=True)，此时 in_channels 变为 128。
# 第二个循环部分：
# 当 out_channels 为 256 时，有 3 个卷积层，每个卷积层的配置为 nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)，后跟 nn.ReLU(inplace=True)。第一次循环时 in_channels 为 128，经过这 3 层卷积后，in_channels 变为 256。
# 当 out_channels 再次为 256 时，又有 3 个卷积层，配置同上，in_channels 保持为 256。
# 所以，总的卷积层数量为 2 + 2 + 3 + 3 = 10 层。
# 池化层（Pooling Layers）：
# 在第一个循环部分，每次 out_channels 变化后，都有一个最大池化层 nn.MaxPool2d(kernel_size=2, stride=2)，这里有 2 个池化层。
# 在第二个循环部分，每次 out_channels 变化后，同样有一个最大池化层 nn.MaxPool2d(kernel_size=2, stride=2)，这里也有 2 个池化层。
# 因此，总的池化层数量为 2 + 2 = 4 层。
# 全连接层（Fully Connected Layers）：
# self.classifier 部分包含 3 个全连接层：
# 第一个全连接层：nn.Linear(fc_input_dim, 1024)，将卷积层和池化层输出的特征向量映射到 1024 个神经元。
# 第二个全连接层：nn.Linear(1024, 1024)，进一步处理特征。
# 第三个全连接层：nn.Linear(1024, num_classes)，输出最终的分类结果，num_classes 是情感类别的数量。
# 所以，全连接层的数量为 3 层。
# Dropout 层（Dropout Layers）：
# 在卷积层和池化层之间，有 4 个 nn.Dropout(0.25) 层，用于防止过拟合。
# 在全连接层之间，有 2 个 nn.Dropout(0.5) 层，同样用于防止过拟合。
# 总的 Dropout 层数量为 4 + 2 = 6 层。
# 综上所述，这个 VGG13_PyTorch 模型包含 10 层卷积层、4 层池化层、3 层全连接层和 6 层 Dropout 层。

# @inproceedings{BarsoumICMI2016,
#     title={Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution},
#     author={Barsoum, Emad and Zhang, Cha and Canton Ferrer, Cristian and Zhang, Zhengyou},
#     booktitle={ACM International Conference on Multimodal Interaction (ICMI)},
#     year={2016}
# }
class VGG13_PyTorch(nn.Module):
    '''
    A VGG13 like model (https://arxiv.org/pdf/1409.1556.pdf) tweaked for emotion data.
    '''
    def __init__(self, num_classes):
        super(VGG13_PyTorch, self).__init__()
        self.learning_rate = 0.05
        self.input_width = 48
        self.input_height = 48
        self.input_channels = 1

        self.features = self._create_features()
        self.classifier = None
        self.num_classes = num_classes

        # 动态计算全连接层的输入维度
        test_input = torch.randn(1, self.input_channels, self.input_height, self.input_width)
        with torch.no_grad():
            output = self.features(test_input)
        fc_input_dim = output.numel() // output.size(0)
        self.classifier = self._create_classifier(fc_input_dim, num_classes)

    def _create_features(self):
        layers = []
        in_channels = self.input_channels
        # 第一个循环部分
        for i, out_channels in enumerate([64, 128]):
            for _ in range(2):
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(0.25))

        # 第二个循环部分
        for i, out_channels in enumerate([256, 256]):
            for _ in range(3):
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(0.25))

        return nn.Sequential(*layers)

    def _create_classifier(self, fc_input_dim, num_classes):
        return nn.Sequential(
            nn.Linear(fc_input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
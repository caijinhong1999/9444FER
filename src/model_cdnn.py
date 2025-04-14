import torch
import torch.nn as nn
# 构建 9 层卷积神经网络
class NineLayerCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(NineLayerCNN, self).__init__()
        self.batch_size = 64
        self.lr = 0.001
        self.epoch = 10
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # ✅ BN 加在卷积后
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.bn4 = nn.BatchNorm1d(128)  # ✅ BN 加在 FC 后
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.relu4(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x


# 构建 12 层卷积神经网络
class TwelveLayerCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(TwelveLayerCNN, self).__init__()
        self.batch_size = 64
        self.lr = 0.001
        self.epoch = 10
        # 第一层卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        # 第二层卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        # 第三层卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        # 第四层卷积块
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256 * 3 * 3, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu5 = nn.ReLU()

        self.fc2 = nn.Linear(256, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.relu6 = nn.ReLU()

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = self.relu5(self.bn5(self.fc1(x)))
        x = self.relu6(self.bn6(self.fc2(x)))
        x = self.fc3(x)
        return x

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
    def __init__(self, num_classes=9):
        super(VGG13_PyTorch, self).__init__()
        self.batch_size = 64
        self.lr = 0.001
        self.epoch = 10
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
                layers.append(nn.BatchNorm2d(out_channels)) # 批量归一化
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(0.25))

        # 第二个循环部分
        for i, out_channels in enumerate([256, 256]):
            for _ in range(3):
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))  # 批量归一化
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(0.25))

        return nn.Sequential(*layers)

    def _create_classifier(self, fc_input_dim, num_classes):
        return nn.Sequential(
            nn.Linear(fc_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

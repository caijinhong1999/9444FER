import torch
import torch.nn as nn
# Define a 9-layer Convolutional Neural Network
class NineLayerCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(NineLayerCNN, self).__init__()
        self.batch_size = 64
        self.lr = 0.001
        self.epoch = 10

        # 6 convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()

        # After 3 pooling layers, the feature map size is 6x6 (assuming input is 48x48)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Block 2
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # Block 3
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)

        x = self.flatten(x)

        # Fully connected layers
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)

        return x


# build 12 layers CNN
class TwelveLayerCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(TwelveLayerCNN, self).__init__()
        self.batch_size = 64
        self.lr = 0.001
        self.epoch = 20

        # Block 1: Conv1 ~ Conv3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout2d(p=0.2)

        # Block 2: Conv4 ~ Conv6
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout2d(p=0.2)

        # Block 3: Conv7 ~ Conv9
        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout2d(p=0.2)

        self.flatten = nn.Flatten()

        # After 3 poolings: 6×6
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.drop_fc1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.drop_fc2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Block 1
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.drop1(x)

        # Block 2
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.drop2(x)

        # Block 3
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.drop3(x)

        x = self.flatten(x)

        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop_fc1(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.drop_fc2(x)
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
        self.epoch = 30
        self.input_width = 48
        self.input_height = 48
        self.input_channels = 1

        self.features = self._create_features()
        self.classifier = None
        self.num_classes = num_classes

        # 1. Dynamically computing the input dimension of fully connected layers
        test_input = torch.randn(1, self.input_channels, self.input_height, self.input_width)
        with torch.no_grad():
            output = self.features(test_input)
        fc_input_dim = output.numel() // output.size(0)
        self.classifier = self._create_classifier(fc_input_dim, num_classes)

    def _create_features(self):
        layers = []
        in_channels = self.input_channels
        # The first loop section
        for i, out_channels in enumerate([64, 128]):
            for _ in range(2):
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(out_channels)) # Batch normalization
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(0.25))

        # The second loop section
        for i, out_channels in enumerate([256, 256]):
            for _ in range(3):
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))  # Batch normalization
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

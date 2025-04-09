import torch.nn as nn


# Create Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, size_change=None):
        super(ResBlock, self).__init__()

        # one convolution layer in each block
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)

        # batch normalization for conv1
        self.bn = nn.BatchNorm2d(out_channel)

        # relu activation function
        self.relu = nn.ReLU(inplace=True)

        # match inter-block dimensions
        self.convert_size = size_change

        # dropout layer to prevent overfitting (random 50%)
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        res = x

        out = self.conv1(x)
        out = self.bn(out)

        # adjust residual if dimensions do not match
        if self.convert_size is not None:
            res = self.convert_size(res)

        # add residual
        out += res

        out = self.relu(out)
        out = self.dropout(out)

        return out


# Create 6 layers Residual Network
class ResNet(nn.Module):
    def __init__(self, classes_num=10):
        super(ResNet, self).__init__()

        # preprocess layer
        self.preprocess = self._make_preprocess_layer()

        # 4 residual layers
        self.layer1 = self._make_layer(in_channel=64, out_channel=64, block_num=1, stride=1)
        self.layer2 = self._make_layer(in_channel=64, out_channel=128, block_num=1, stride=2, check_schange=True)
        self.layer3 = self._make_layer(in_channel=128, out_channel=256, block_num=1, stride=2, check_schange=True)
        self.layer4 = self._make_layer(in_channel=256, out_channel=512, block_num=1, stride=2, check_schange=True)

        # classification layer
        self.classifier = self._make_classifier(classes_num)

    def _make_preprocess_layer(self):
        return nn.Sequential(
            # convert input into residual processing format
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def _make_layer(self, in_channel, out_channel, block_num, stride, check_schange=False):
        size_change = None

        # adjust input size for inter-block value
        if check_schange:
            size_change = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

        # add residual blocks to a list
        layers = []
        layers.append(ResBlock(in_channel, out_channel, stride, size_change))

        for _ in range(1, block_num):
            layers.append(ResBlock(out_channel, out_channel))

        return nn.Sequential(*layers)

    def _make_classifier(self, classes_num):
        return nn.Sequential(
            # global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # fully connected classifier layers (3)
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, classes_num)
        )

    def forward(self, x):
        x = self.preprocess(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x


# Create 10 layers Residual Network
class ResNetEX(nn.Module):
    def __init__(self, classes_num=10):
        super(ResNetEX, self).__init__()

        # preprocess layer
        self.preprocess = self._make_preprocess_layer()

        # 8 residual layers
        self.layer1 = self._make_layer(in_channel=64, out_channel=64, block_num=2, stride=1)
        self.layer2 = self._make_layer(in_channel=64, out_channel=128, block_num=2, stride=2, check_schange=True)
        self.layer3 = self._make_layer(in_channel=128, out_channel=256, block_num=2, stride=2, check_schange=True)
        self.layer4 = self._make_layer(in_channel=256, out_channel=512, block_num=2, stride=2, check_schange=True)

        # classification layer
        self.classifier = self._make_classifier(classes_num)

    def _make_preprocess_layer(self):
        return nn.Sequential(
            # convert input into residual processing format
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def _make_layer(self, in_channel, out_channel, block_num, stride, check_schange=False):
        size_change = None

        # adjust input size for inter-block value
        if check_schange:
            size_change = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

        # add residual blocks to a list
        layers = []
        layers.append(ResBlock(in_channel, out_channel, stride, size_change))

        for _ in range(1, block_num):
            layers.append(ResBlock(out_channel, out_channel))

        return nn.Sequential(*layers)

    def _make_classifier(self, classes_num):
        return nn.Sequential(
            # global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # fully connected classifier layers (3)
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, classes_num)
        )

    def forward(self, x):
        x = self.preprocess(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x

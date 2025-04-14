# ResNet18
class ResNet(nn.Module):
    def __init__(self, classes_num=9):
        super(ResNet, self).__init__()
        self.batch_size = 64
        self.lr = 0.001
        self.epoch = 10
        
        # load pre-trained ResNet18 model
        self.model = models.resnet18(weights=None)

        # edit input channel in pre-process layer
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # edit final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, classes_num)
    
    def forward(self, x):
        x = self.model(x)
        return x





# ResNet18 + CBAM
class channel_attention(nn.Module):
    def __init__(self, in_channel, reduce=8):
        super(channel_attention, self).__init__()
        # max/avg pooling
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.max_lin1 = nn.Linear(in_channel, in_channel // reduce, bias=False)
        self.max_lin2 = nn.Linear(in_channel // reduce, in_channel, bias=False)
        self.avg_lin1 = nn.Linear(in_channel, in_channel // reduce, bias=False)
        self.avg_lin2 = nn.Linear(in_channel // reduce, in_channel, bias=False)

        self.dropout = nn.Dropout(p=0.2)

        # activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # get x size
        b, c, _, _ = x.size()
        
        # max pooling process
        max_output = self.max_pooling(x).view(b, c)
        max_output = self.max_lin1(max_output)
        max_output = self.relu(max_output)
        max_output = self.dropout(max_output)
        max_output = self.max_lin2(max_output)
        

        # avg pooling process
        avg_output = self.avg_pooling(x).view(b, c)
        avg_output = self.avg_lin1(avg_output)
        avg_output = self.relu(avg_output)
        avg_output = self.dropout(avg_output)
        avg_output = self.avg_lin2(avg_output)

        # add attention
        output = max_output + avg_output

        # activate
        output = self.sigmoid(output)
        
        output = output.view(b, c, 1, 1)

        return output

class spatial_attention(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_output, _ = torch.max(x, dim=1, keepdim=True)
        avg_output = torch.mean(x, dim=1, keepdim=True)

        # add avg and max result
        output = torch.cat([max_output, avg_output], dim=1)

        output = self.conv(output)

        output = self.sigmoid(output)

        return output

class CBAM(nn.Module):
    def __init__(self, in_channel, reduce=8, kernel_size=3):
        super(CBAM, self).__init__()
        self.channel_attention = channel_attention(in_channel, reduce)
        self.spatial_attention = spatial_attention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x + x
        out = self.spatial_attention(out) * out + out
        return out

class residual_block_CBAM(BasicBlock):
    def __init__(self, *args, **kwargs):
        super(residual_block_CBAM, self).__init__(*args, **kwargs)
        self.cbam = CBAM(self.conv2.out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)

        # add CBAM block
        output = self.cbam(output)

        # residual connection
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        output += residual
        
        output = self.relu(output)

        return output

class ResNet_CBAM(nn.Module):
    def __init__(self, classes_num=9):
        super(ResNet_CBAM, self).__init__()
        self.batch_size = 64
        self.lr = 0.001
        self.epoch = 10
        
        self.model = ResNet(block=residual_block_CBAM, layers=[2, 2, 2, 2])

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.model.fc = nn.Linear(self.model.fc.in_features, classes_num)
    
    def forward(self, x):
        x = self.model(x)
        return x

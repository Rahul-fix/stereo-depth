from torch import nn

class Siamese(nn.Module):
    def __init__(self, block, in_channels, channels,kernel_size_res,kernel_size_siamese, num_blocks):
        super(Siamese, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.block = block
        self.num_blocks = num_blocks
        self.conv1 = nn.Conv2d(self.in_channels,self.channels,kernel_size_siamese,stride=1,padding=3)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        layers = list()
        layers.append(self.block(channels,kernel_size_res,stride=2,padding=1,change_dim=True))
        for i in range(1,num_blocks):
            layers.append(self.block(channels,kernel_size_res,stride=1,padding=1))
        self.residual_sequence = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.residual_sequence(x)
        return x

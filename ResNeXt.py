import torch
import torch.nn as nn
print(torch.__version__)

# A cardianlity block in ResNeXt50 has 32 similar elements. 
# Each element contains Conv2D(1x1) -> Conv2D(3x3) -> Conv2D(1x1)
class CardinalityBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample = None, stride = 1, C = 32):
        super(CardinalityBlock, self).__init__()
        self.expansion = 2 
        self.small_out_channels = out_channels // C # actual number of out channels for each element.
        self.conv1 = nn.Conv2d(in_channels, self.small_out_channels, kernel_size = 1, stride = 1, padding = 0)
        self.bn1 = nn.BatchNorm2d(self.small_out_channels)
        self.conv2 = nn.Conv2d(self.small_out_channels, self.small_out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn2 = nn.BatchNorm2d(self.small_out_channels)
        self.conv3 = nn.Conv2d(self.small_out_channels, self.small_out_channels * self.expansion, kernel_size = 1, stride = 1, padding = 0)
        self.bn3 = nn.BatchNorm2d(self.small_out_channels * self.expansion)
        self.relu = nn.ReLU()
        # pack 3 conv blocks into a whole vertical branch
        self.branch = nn.Sequential(self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3) 
        self.identity_downsample = identity_downsample
        self.C = C
        
    
    def forward(self, x):
        # get information from previous cardinality block
        identity = x
        branch_list = []

        # concat all 32 branches
        for i in range(self.C):
            branch_list.append(self.branch(x))
        
        x = torch.cat(branch_list, 1)

        # add identity information when it is not None (after a whole branch)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x

# ResNeXt architecture 
class ResNeXt(nn.Module):
    def __init__(self, cardinalityBlock, num_repeat, image_channels, num_classes):
        super(ResNeXt, self).__init__()
        # Before ResBlock
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        # ResBlock. For ResNeXt50, num_repeat = [3, 4, 6, 3]
        self.conv2 = self.create_resBlock(cardinalityBlock, num_repeat[0], out_channels=128, stride=1)
        self.conv3 = self.create_resBlock(cardinalityBlock, num_repeat[1], out_channels=256, stride=2)
        self.conv4 = self.create_resBlock(cardinalityBlock, num_repeat[2], out_channels=512, stride=2)
        self.conv5 = self.create_resBlock(cardinalityBlock, num_repeat[3], out_channels=1024, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024 * 2, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


    def create_resBlock(self, cardinalityBlock, num_blocks, out_channels, stride):
        identity_downsample = None
        conv_layers = []

        # Only apply identity when changing to the new conv layer 
        if self.in_channels != out_channels * 2:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels * 2, kernel_size = 1, stride = stride),
                                                nn.BatchNorm2d(out_channels * 2))
        conv_layers.append(cardinalityBlock(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 2 # expansion rate of 2

        for i in range(num_blocks - 1):
            conv_layers.append(cardinalityBlock(self.in_channels, out_channels))
        return nn.Sequential(*conv_layers)
            
def ResNeXt50(image_channels = 3, num_classes = 1000):
    return ResNeXt(CardinalityBlock, [3, 4, 6, 3], image_channels, num_classes)

if __name__ == '__main__':
    x = torch.randn(3, 3, 224, 224)
    model = ResNeXt50()
    print(model(x).shape) # torch.Size([3, 1000])

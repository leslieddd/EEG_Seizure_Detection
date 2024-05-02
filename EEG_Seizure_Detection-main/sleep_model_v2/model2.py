import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pool, pool_type='max', dropout_prob=0.5):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=kernel_size//2)

        if pool_type == 'max':
            self.pool = nn.MaxPool2d(pool, stride=pool[1])
        else:
            self.pool = nn.AvgPool2d(pool, stride=pool[1])

        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

# class SleepStageCNN(nn.Module):
#     def __init__(self):
#         super(SleepStageCNN, self).__init__()
#         # Modify the input channel for the first block from 18 to 4
#         self.block1 = ConvBlock(4, 32, 3, 1, (2, 2), 'max', 0.5)
#         self.block2 = ConvBlock(32, 32, 3, 1, (2, 2), 'avg', 0.5)
#         self.block3 = ConvBlock(32, 64, 3, 1, (2, 2), 'max', 0.5)
#         self.block4 = ConvBlock(64, 64, 3, 1, (2, 2), 'avg', 0.5)
#         self.block5 = ConvBlock(64, 64, 3, 1, (2, 2), 'max', 0.5)
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#
#     def forward(self, x):
#         x = self.block1(x)
#         identity1 = x
#         x = self.block2(x)
#
#         identity1 = F.interpolate(identity1, size=(x.size(2), x.size(3)), mode='nearest')
#         x += identity1  # Residual connection after block 2
#
#         x = self.block3(x)
#         identity2 = x
#         x = self.block4(x)
#
#         identity2 = F.interpolate(identity2, size=(x.size(2), x.size(3)), mode='nearest')
#         x += identity2  # Residual connection after block 4
#
#         x = self.block5(x)
#         x = self.gap(x)
#         x = torch.flatten(x, 1)
#         return x

class SleepStageCNN(nn.Module):
    def __init__(self):
        super(SleepStageCNN, self).__init__()
        # 修改第一个卷积块的输入通道数从4改为18
        self.block1 = ConvBlock(18, 32, 3, 1, (2, 2), 'max', 0.5)
        self.block2 = ConvBlock(32, 32, 3, 1, (2, 2), 'avg', 0.5)
        self.block3 = ConvBlock(32, 64, 3, 1, (2, 2), 'max', 0.5)
        self.block4 = ConvBlock(64, 64, 3, 1, (2, 2), 'avg', 0.5)
        self.block5 = ConvBlock(64, 64, 3, 1, (2, 2), 'max', 0.5)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.block1(x)
        identity1 = x
        x = self.block2(x)

        identity1 = F.interpolate(identity1, size=(x.size(2), x.size(3)), mode='nearest')
        x += identity1  # 残差连接

        x = self.block3(x)
        identity2 = x
        x = self.block4(x)

        identity2 = F.interpolate(identity2, size=(x.size(2), x.size(3)), mode='nearest')
        x += identity2  # 残差连接

        x = self.block5(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return x



class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BinaryClassifier, self).__init__()
        # Define the MLP layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First dense layer
        self.fc2 = nn.Linear(hidden_dim, 1)          # Output layer for binary classification

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation function
        x = self.fc2(x)          # Output logits for binary classification
        x = torch.sigmoid(x)     # Apply sigmoid to output probabilities
        return x

class EEGSNet(nn.Module):
    def __init__(self):
        super(EEGSNet, self).__init__()
        self.cnn = SleepStageCNN()  # Assuming SleepStageCNN is defined as above
        self.classifier = BinaryClassifier(input_dim=64, hidden_dim=128)  # Initialize the classifier with 128 hidden units

    def forward(self, x):
        x = self.cnn(x)           # Get the CNN features
        x = self.classifier(x)    # Classify the features into two classes
        return x

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Initialize model
# model = EEGSNet().to(device)
# input_tensor = torch.randn(10, 4, 129, 111)  # Example input
# output = model(input_tensor)
# print(output.shape)  # Should now output the shape [1, 4], each summing to 1
# print(output)  # Outputs the probabilities for each class
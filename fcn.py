import torch
import torch.nn as nn
import torch.optim as optim

class DoubleConv(nn.Module):
    """ (卷积 => BN => ReLU) * 2 """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SimpleUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SimpleUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.up1 = DoubleConv(128 + 64, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = nn.MaxPool2d(2)(x1)
        x2 = self.down1(x2)
        x = nn.functional.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x1, x], dim=1)
        x = self.up1(x)
        logits = self.outc(x)
        return logits

# 模型初始化
model = SimpleUNet(n_channels=3, n_classes=3)  # 假设输入是RGB图像，有3个类别
print(model)

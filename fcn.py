import torch
import torch.nn as nn
import torch.optim as optim
from attention_gate import attention_gate
class DoubleConv(nn.Module):
    """ (卷积 => BN => ReLU => Dropout) * 2 """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.double_conv(x)

class MyUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(MyUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        #self.att1 = attention_gate(F_g=128, F_l=64, F_int=32, F_out=64)
        self.up1 = DoubleConv(128 + 64, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = nn.MaxPool2d(2)(x1)
        x2 = self.down1(x2)
        x = nn.functional.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        #x = self.att1(g=x, x=x1)  # Apply attention gate
        x = torch.cat([x1, x], dim=1)
        x = self.up1(x)
        logits = self.outc(x)
        return logits

# 模型初始化
model = MyUNet(n_channels=3, n_classes=3)  # 假设输入是RGB图像，有3个类别
print(model)

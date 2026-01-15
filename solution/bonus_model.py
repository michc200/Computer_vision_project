"""Define your architecture here."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidual(nn.Module):
    """MobileNetV2 Inverted Residual Block."""
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.extend([
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise linear projection
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class BonusModel(nn.Module):
    """Efficient MobileNetV2-inspired model with conv3 for grad-cam compatibility."""
    def __init__(self, num_classes=2):
        super(BonusModel, self).__init__()

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )

        # Series of efficient Inverted Residual blocks
        self.block1 = InvertedResidual(16, 24, 2, 6)   # 128x128 -> 64x64
        self.block2 = InvertedResidual(24, 32, 2, 6)   # 64x64 -> 32x32

        # IMPORTANT: Named as conv3 for grad-cam compatibility
        self.conv3 = nn.Sequential(
            InvertedResidual(32, 64, 2, 6),   # 32x32 -> 16x16
            InvertedResidual(64, 96, 1, 6),   # 16x16 -> 16x16
        )

        self.block5 = InvertedResidual(96, 160, 2, 6)  # 16x16 -> 8x8

        # Final conv
        self.final_conv = nn.Sequential(
            nn.Conv2d(160, 320, 1, 1, 0, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU6(inplace=True)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(320, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.conv3(x)  # Target for Grad-CAM
        x = self.block5(x)
        x = self.final_conv(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def my_bonus_model():
    """Override the model initialization here."""
    model = BonusModel()
    # Ensure this path matches the one in bonus_main.py
    model.load_state_dict(torch.load('checkpoints/bonus.pt')['model'])
    return model
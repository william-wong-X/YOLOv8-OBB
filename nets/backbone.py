import torch
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # 自动计算padding值
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # 标准卷积Conv(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)

    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # 对输入张量进行卷积、批量归一化和激活
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        # 删掉批量归一化
        return self.act(self.conv(x))

class SPPF(nn.Module):
    # Glenn Jocher 为 YOLOv5 设计的空间金字塔池化--快速 (SPPF) 层

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # 隐藏层的通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

class Bottleneck(nn.Module):
    # 标准的bottleneck

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 隐藏层的通道数
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    # 用 2 次卷积更快地实现CSP Bottleneck

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # 隐藏层的通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 可选act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        # 使用 split() 代替 chunk() 向前传递
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class Backbone(nn.Module):
    def __init__(self, 
                 base_channels: int, 
                 base_depth: int):
        super(Backbone, self).__init__()
        #-----------------------------------------------#
        #   输入图片是3, 640, 640
        #-----------------------------------------------#
        # 3, 640, 640 -> 64, 320, 320
        self.stem = Conv(3, base_channels, 3, 2)
        
        # 64, 320, 320 -> 128, 160, 160 -> 128, 160, 160
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f(base_channels * 2, base_channels * 2, base_depth, True),
        )
        # 128, 160, 160 -> 256, 80, 80 -> 256, 80, 80
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        # 256, 80, 80 -> 512, 40, 40 -> 512, 40, 40
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )
        # 512, 40, 40 -> 1024, 20, 20 -> 1024, 20, 20
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2),
            C2f(base_channels * 16, base_channels * 16, base_depth, True),
            SPPF(base_channels * 16, base_channels * 16, k=5)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        #-----------------------------------------------#
        #   dark3的输出为256, 80, 80，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        #-----------------------------------------------#
        #   dark4的输出为512, 40, 40，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        #-----------------------------------------------#
        #   dark5的输出为1024 * deep_mul, 20, 20，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3
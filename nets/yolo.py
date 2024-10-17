import math
import sys

import torch
import torch.nn as nn

sys.path.append('D:\program\python\pytorch\yolov8-obb')
from nets.backbone import Backbone, C2f, Conv
from utils.torch_utils import fuse_conv_and_bn, weights_init
from utils.tal import make_anchors

def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 phi: str, 
                 ne: int = 1):
        super(YoloBody, self).__init__()
        depth_dict          = {'n' : 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.00,}
        width_dict          = {'n' : 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul = depth_dict[phi], width_dict[phi]
        
        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3
        #--------------------------------------------------------------------------#
        #   输入图片是3, 640, 640 -> backbone输出256，80，80 512，40，40 1024，20，20
        #--------------------------------------------------------------------------#
        self.backbone   = Backbone(base_channels, base_depth)
        
        #------------------------加强特征提取网络------------------------# 
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        # 1024 + 512, 40, 40 -> 512, 40, 40
        self.conv3_for_upsample1    = C2f(base_channels * 16 + base_channels * 8, base_channels * 8, base_depth, shortcut=False)
        # 768, 80, 80 -> 256, 80, 80
        self.conv3_for_upsample2    = C2f(base_channels * 8 + base_channels * 4, base_channels * 4, base_depth, shortcut=False)
        
        # 256, 80, 80 -> 256, 40, 40
        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        # 512 + 256, 40, 40 -> 512, 40, 40
        self.conv3_for_downsample1  = C2f(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth, shortcut=False)

        # 512, 40, 40 -> 512, 20, 20
        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        # 1024 + 512, 20, 20 ->  1024, 20, 20
        self.conv3_for_downsample2  = C2f(base_channels * 16 + base_channels * 8, base_channels * 16, base_depth, shortcut=False)
        #------------------------加强特征提取网络------------------------#
        
        #---------------------------检测头网络---------------------------#
        ch = [base_channels * 4, base_channels * 8, base_channels * 16]
        self.shape = None
        self.nc = num_classes  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = num_classes + self.reg_max * 4  # number of outputs per anchor
        self.ne = ne  # number of extra parameters
        # self.stride = torch.zeros(self.nl)  # strides computed during build
        self.stride     = torch.tensor([256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))])  # forward
        c2, c3, c4 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100)), max(ch[0] // 4, self.ne)  # channels
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)
        weights_init(self)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        #---------------------------检测头网络---------------------------#
    
    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self
    
    def forward(self, x):
        # backbone
        feat1, feat2, feat3 = self.backbone.forward(x)
        
        #------------------------加强特征提取网络------------------------# 
        # 1024, 20, 20 -> 1024 * deep_mul, 40, 40
        P5_upsample = self.upsample(feat3)
        # 1024, 40, 40 cat 512, 40, 40 -> 1024 * deep_mul + 512, 40, 40
        P4          = torch.cat([P5_upsample, feat2], 1)
        # 1024 + 512, 40, 40 -> 512, 40, 40
        P4          = self.conv3_for_upsample1(P4)

        # 512, 40, 40 -> 512, 80, 80
        P4_upsample = self.upsample(P4)
        # 512, 80, 80 cat 256, 80, 80 -> 768, 80, 80
        P3          = torch.cat([P4_upsample, feat1], 1)
        # 768, 80, 80 -> 256, 80, 80
        P3          = self.conv3_for_upsample2(P3)

        # 256, 80, 80 -> 256, 40, 40
        P3_downsample = self.down_sample1(P3)
        # 512, 40, 40 cat 256, 40, 40 -> 768, 40, 40
        P4 = torch.cat([P3_downsample, P4], 1)
        # 768, 40, 40 -> 512, 40, 40
        P4 = self.conv3_for_downsample1(P4)

        # 512, 40, 40 -> 512, 20, 20
        P4_downsample = self.down_sample2(P4)
        # 512, 20, 20 cat 1024, 20, 20 -> 1024 + 512, 20, 20
        P5 = torch.cat([P4_downsample, feat3], 1)
        # 1024 + 512, 20, 20 -> 1024, 20, 20
        P5 = self.conv3_for_downsample2(P5)
        #------------------------加强特征提取网络------------------------# 
        # P3 256, 80, 80
        # P4 512, 40, 40
        # P5 1024, 20, 20
        shape = P3.shape  # BCHW
        x = [P3, P4, P5]
        x1 = [P3, P4, P5]
        #---------------------------检测头网络---------------------------#
        bs = x[0].shape[0]  # batch size
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        # x[0]: num_classes + self.reg_max * 4, 80, 80
        # x[1]: num_classes + self.reg_max * 4, 40, 50
        # x[2]: num_classes + self.reg_max * 4, 20, 20
        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = self.dfl(box)
        angle = torch.cat([self.cv4[i](x1[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        #---------------------------检测头网络---------------------------#
        return dbox, cls, x, angle, self.anchors.to(dbox.device), self.strides.to(dbox.device)

# # 测试
# if __name__ == '__main__':
#     x = torch.randn(1, 3, 640, 640)
#     m = YoloBody(7, 's')
#     out = m(x)
#     print(out)
#     print(out[0].shape)
#     print(out[1].shape)
#     print(out[2][1].shape)
#     print(out[3].shape)
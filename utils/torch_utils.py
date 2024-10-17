import math
from copy import deepcopy

import torch
from torch import nn

def fuse_conv_and_bn(conv, bn):
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
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

def copy_attr(a, b, include=(), exclude=()):
    # 将对象 “b ”中的属性复制到对象 “a ”中，并提供包含/排除某些属性的选项
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)

def is_parallel(model):
    # 如果模型属于 DP 或 DDP 类型，则返回 True
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def de_parallel(model):
    # 去并行化模型：如果模型属于 DP 或 DDP 类型，则返回单 GPU 模型
    return model.module if is_parallel(model) else model

class ModelEMA:
    """更新的指数移动平均 (EMA) 来自 https://github.com/rwightman/pytorch-image-models 
    保持模型 state_dict（参数和缓冲区）中所有内容的移动平均值 有关 EMA 的详细信息，
    请参阅 https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage 
    要禁用 EMA，请将 `enabled` 属性设为 `False`。
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # 创建 EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # EMA 更新次数
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # 衰变指数斜坡 (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        # 更新 EMA 参数
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # true for FP16 and FP32
                    v *= d
                    v += (1 - d) * msd[k].detach()
                    # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype},  model {msd[k].dtype}'

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        # 更新属性并保存已删除优化器的剥离模型
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)

def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)
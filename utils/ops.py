import torch
import cv2
import numpy as np

def xyxy2xywh(x):
    """
    将边界框坐标从 (x1, y1, x2, y2) 格式转换为 (x, y, width, height) 格式，其中 (x1, y1) 为左上角，(x2, y2) 为右下角。为左上角，(x2, y2) 为右下角.

    Args:
        x (np.ndarray | torch.Tensor): 输入的边界框坐标，格式为 (x1, y1, x2, y2).

    Returns:
        y (np.ndarray | torch.Tensor): 边界框坐标，格式为 (x, y, width, height).
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    """
    将边界框坐标从 (x, y, width, height) 格式转换为 (x1, y1, x2, y2) 格式，其中 (x1, y1) 为左上角，(x2, y2) 为右下角。为左上角，(x2, y2) 为右下角。
    注意：每 2 个通道的运行速度比每个通道快.

    Args:
        x (np.ndarray | torch.Tensor): 输入的边界框坐标，格式为 (x, y, width, height).

    Returns:
        y (np.ndarray | torch.Tensor): 边界框坐标，格式为 (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y

def xyxyxyxy2xywhr(x):
    """
    将[xy1, xy2, xy3, xy4]分批定向边框 (OBB) 转换为[xywh, rotation]。旋转值的单位为 0 至 90 度.

    Args:
        x (numpy.ndarray | torch.Tensor): 形状 (n, 8) 的输入框角 [xy1, xy2, xy3, xy4].

    Returns:
        (numpy.ndarray | torch.Tensor): 以 [cx、cy、w、h、rotation] 格式转换形状 (n, 5) 的数据.
    """
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().numpy() if is_torch else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        # NOTE: Use cv2.minAreaRect to get accurate xywhr,
        # especially some objects are cut off by augmentations in dataloader.
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([cx, cy, w, h, angle / 180 * np.pi])
    return torch.tensor(rboxes, device=x.device, dtype=x.dtype) if is_torch else np.asarray(rboxes)


def xywhr2xyxyxyxy(x):
    """
    将 [xywh, rotation] 的批次定向边框 (OBB) 转换为 [xy1, xy2, xy3, xy4]。旋转值的单位应为 0 至 90 度.

    Args:
        x (numpy.ndarray | torch.Tensor): 形状为 (n, 5) 或 (b, n, 5) 的 [cx, cy, w, h, rotation] 格式方框.

    Returns:
        (numpy.ndarray | torch.Tensor): 形状 (n, 4, 2) 或 (b, n, 4, 2) 的转换角点.
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)
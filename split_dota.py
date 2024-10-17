import itertools
from glob import glob
from math import ceil
from pathlib import Path
import contextlib
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from shapely.geometry import Polygon

class_flie = ''

def img2label_paths(img_paths):
    # 将标签路径定义为图像路径的函数
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]

def exif_size(img: Image.Image):
    # 返回经 exif 校正的 PIL 尺寸
    s = img.size  # (width, height)
    if img.format == "JPEG":  # 只支持 JPEG images
        with contextlib.suppress(Exception):
            exif = img.getexif()
            if exif:
                rotation = exif.get(274, None)  # 方位标签的 EXIF 键为 274
                if rotation in {6, 8}:  # 旋转 270 或 90
                    s = s[1], s[0]
    return s

def bbox_iof(polygon1, bbox2, eps=1e-6):
    """
    计算 bbox1 和 bbox2 的 iofs.
    iof: (bbox1 ∩ bbox2) / bbox1

    Args:
        polygon1 (np.ndarray): polygon 坐标, (n, 8).
        bbox2 (np.ndarray): 边界框, (n ,4).
    """
    polygon1 = polygon1.reshape(-1, 4, 2)
    lt_point = np.min(polygon1, axis=-2)  # left-top
    rb_point = np.max(polygon1, axis=-2)  # right-bottom
    bbox1 = np.concatenate([lt_point, rb_point], axis=-1)

    lt = np.maximum(bbox1[:, None, :2], bbox2[..., :2])
    rb = np.minimum(bbox1[:, None, 2:], bbox2[..., 2:])
    wh = np.clip(rb - lt, 0, np.inf)
    h_overlaps = wh[..., 0] * wh[..., 1]

    left, top, right, bottom = (bbox2[..., i] for i in range(4))
    polygon2 = np.stack([left, top, right, top, right, bottom, left, bottom], axis=-1).reshape(-1, 4, 2)

    sg_polys1 = [Polygon(p) for p in polygon1]
    sg_polys2 = [Polygon(p) for p in polygon2]
    overlaps = np.zeros(h_overlaps.shape)
    for p in zip(*np.nonzero(h_overlaps)):
        overlaps[p] = sg_polys1[p[0]].intersection(sg_polys2[p[-1]]).area
    unions = np.array([p.area for p in sg_polys1], dtype=np.float32)
    unions = unions[..., None]

    unions = np.clip(unions, eps, np.inf)
    outputs = overlaps / unions
    if outputs.ndim == 1:
        outputs = outputs[..., None]
    return outputs


def load_yolo_dota(data_root, split="train"):
    """
    Load DOTA dataset.

    Args:
        data_root (str): Data root.
        split (str): 分割数据集，可以是训练数据集，也可以是评估数据集.

    Notes:
        为 DOTA 数据集假定的目录结构:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    class_name = []
    with open(class_flie, 'r') as file:
        for line in file:
            name = line.strip()
            class_name.append(name)
    assert split in {"train", "val"}, f"Split must be 'train' or 'val', not {split}."
    im_dir = Path(data_root) / "images" / split
    assert im_dir.exists(), f"Can't find {im_dir}, please check your data root."
    im_files = glob(str(Path(data_root) / "images" / split / "*"))
    lb_files = img2label_paths(im_files)
    annos = []
    for im_file, lb_file in zip(im_files, lb_files):
        w, h = exif_size(Image.open(im_file))
        with open(lb_file) as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            # GLH-Bridge数据集读取
            for i in range(len(lb)):
                lb[i] = [class_name.index(lb[i][8])] + lb[i][:-2]
            lb = np.array(lb, dtype=np.float32)
        annos.append(dict(ori_size=(h, w), label=lb, filepath=im_file))
    return annos


def get_windows(im_size, crop_sizes=(1024,), gaps=(200,), im_rate_thr=0.6, eps=0.01):
    """
    获取窗口坐标.

    Args:
        im_size (tuple): 原始图片大小, (h, w).
        crop_sizes (List(int)): 分割窗的大小.
        gaps (List(int)): 分割窗的步长.
        im_rate_thr (float): 窗口区域除以图像区域的阈值.
        eps (float): 数学运算的 Epsilon 值.
    """
    h, w = im_size
    windows = []
    for crop_size, gap in zip(crop_sizes, gaps):
        assert crop_size > gap, f"invalid crop_size gap pair [{crop_size} {gap}]"
        step = crop_size - gap

        xn = 1 if w <= crop_size else ceil((w - crop_size) / step + 1)
        xs = [step * i for i in range(xn)]
        if len(xs) > 1 and xs[-1] + crop_size > w:
            xs[-1] = w - crop_size

        yn = 1 if h <= crop_size else ceil((h - crop_size) / step + 1)
        ys = [step * i for i in range(yn)]
        if len(ys) > 1 and ys[-1] + crop_size > h:
            ys[-1] = h - crop_size

        start = np.array(list(itertools.product(xs, ys)), dtype=np.int64)
        stop = start + crop_size
        windows.append(np.concatenate([start, stop], axis=1))
    windows = np.concatenate(windows, axis=0)

    im_in_wins = windows.copy()
    im_in_wins[:, 0::2] = np.clip(im_in_wins[:, 0::2], 0, w)
    im_in_wins[:, 1::2] = np.clip(im_in_wins[:, 1::2], 0, h)
    im_areas = (im_in_wins[:, 2] - im_in_wins[:, 0]) * (im_in_wins[:, 3] - im_in_wins[:, 1])
    win_areas = (windows[:, 2] - windows[:, 0]) * (windows[:, 3] - windows[:, 1])
    im_rates = im_areas / win_areas
    if not (im_rates > im_rate_thr).any():
        max_rate = im_rates.max()
        im_rates[abs(im_rates - max_rate) < eps] = 1
    return windows[im_rates > im_rate_thr]


def get_window_obj(anno, windows, iof_thr=0.7):
    # 获取每个窗口的对象
    h, w = anno["ori_size"]
    label = anno["label"]
    if len(label):
        # label[:, 1::2] *= w
        # label[:, 2::2] *= h
        iofs = bbox_iof(label[:, 1:], windows)
        # 非规范化和错位坐标
        return [(label[iofs[:, i] >= iof_thr]) for i in range(len(windows))]  # window_anns
    else:
        return [np.zeros((0, 9), dtype=np.float32) for _ in range(len(windows))]  # window_anns


def crop_and_save(anno, windows, window_objs, im_dir, lb_dir):
    """
    裁剪图像并保存新标签.

    Args:
        anno (dict): 注释列表, including `filepath`, `label`, `ori_size` as its keys.
        windows (list): 窗口坐标列表.
        window_objs (list): 每个窗口内的标签列表.
        im_dir (str): 图像的输出目录路径.
        lb_dir (str): 标签的输出目录路径.

    Notes:
        为 DOTA 数据集假定的目录结构:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    im = cv2.imread(anno["filepath"])
    name = Path(anno["filepath"]).stem
    for i, window in enumerate(windows):
        x_start, y_start, x_stop, y_stop = window.tolist()
        new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"
        patch_im = im[y_start:y_stop, x_start:x_stop]
        ph, pw = patch_im.shape[:2]

        cv2.imwrite(str(Path(im_dir) / f"{new_name}.jpg"), patch_im)
        label = window_objs[i]
        if len(label) == 0:
            continue
        label[:, 1::2] -= x_start
        label[:, 2::2] -= y_start
        # label[:, 1::2] /= pw
        # label[:, 2::2] /= ph

        with open(Path(lb_dir) / f"{new_name}.txt", "w") as f:
            for lb in label:
                formatted_coords = ["{:.6g}".format(coord) for coord in lb[1:]]
                f.write(f"{int(lb[0])} {' '.join(formatted_coords)}\n")


def split_images_and_labels(data_root, save_dir, split="train", crop_sizes=(1024,), gaps=(200,)):
    """
    分割图像和标签.

    Notes:
        DOTA 数据集假定的目录结构为:
            - data_root
                - images
                    - split
                - labels
                    - split
        输出目录结构为:
            - save_dir
                - images
                    - split
                - labels
                    - split
    """
    im_dir = Path(save_dir) / "images" / split
    im_dir.mkdir(parents=True, exist_ok=True)
    lb_dir = Path(save_dir) / "labels" / split
    lb_dir.mkdir(parents=True, exist_ok=True)

    annos = load_yolo_dota(data_root, split=split)
    for anno in tqdm(annos, total=len(annos), desc=split):
        windows = get_windows(anno["ori_size"], crop_sizes, gaps)
        window_objs = get_window_obj(anno, windows)
        crop_and_save(anno, windows, window_objs, str(im_dir), str(lb_dir))


def split_trainval(data_root, save_dir, crop_size=1024, gap=200, rates=(1.0,)):
    """
    分割DOTA的训练集和评估集.

    Notes:
        DOTA 数据集假定的目录结构为:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
        输出目录结构为:
            - save_dir
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    crop_sizes, gaps = [], []
    for r in rates:
        crop_sizes.append(int(crop_size / r))
        gaps.append(int(gap / r))
    for split in ["train", "val"]:
        split_images_and_labels(data_root, save_dir, split, crop_sizes, gaps)


def split_test(data_root, save_dir, crop_size=1024, gap=200, rates=(1.0,)):
    """
    DOTA 分离测试集，本测试集不含标签.

    Notes:
        DOTA 数据集假定的目录结构为:
            - data_root
                - images
                    - test
        输出目录结构为:
            - save_dir
                - images
                    - test
    """
    crop_sizes, gaps = [], []
    for r in rates:
        crop_sizes.append(int(crop_size / r))
        gaps.append(int(gap / r))
    save_dir = Path(save_dir) / "images" / "test"
    save_dir.mkdir(parents=True, exist_ok=True)

    im_dir = Path(data_root) / "images" / "test"
    assert im_dir.exists(), f"Can't find {im_dir}, please check your data root."
    im_files = glob(str(im_dir / "*"))
    for im_file in tqdm(im_files, total=len(im_files), desc="test"):
        w, h = exif_size(Image.open(im_file))
        windows = get_windows((h, w), crop_sizes=crop_sizes, gaps=gaps)
        im = cv2.imread(im_file)
        name = Path(im_file).stem
        for window in windows:
            x_start, y_start, x_stop, y_stop = window.tolist()
            new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"
            patch_im = im[y_start:y_stop, x_start:x_stop]
            cv2.imwrite(str(save_dir / f"{new_name}.jpg"), patch_im)


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = 400000000  # 用于解决图片过大的报错
    split_trainval(data_root="yolov8-obb/DOTA_dataset", save_dir="yolov8-obb/DOTA_dataset_split", rates=(0.8,1.0,1.6))
    split_test(data_root="yolov8-obb/DOTA_dataset", save_dir="yolov8-obb/DOTA_dataset_split")

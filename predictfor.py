import os

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

from yolo import YOLO

if __name__ == "__main__":
    # 加载模型
    yolo = YOLO()
    
    crop = False    # crop 指定了是否在单张图片预测后对目标进行截取
    count = False   # count 指定了是否进行目标的计数
    
    # 源文件夹和目标文件夹路径
    src_folder = "D:\\ship_detection_online\\testImages"
    dst_folder = "D:\\testimage\\01"

    # 创建目标文件夹（如果不存在）
    os.makedirs(dst_folder, exist_ok=True)
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # 使用tqdm创建进度条
    for filename in tqdm(image_files, desc="Processing images"):
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)
        try:
            image = Image.open(src_path)
        except:
            print(f'Open Error! Try again! File: {filename}')
            continue
        else:
            r_image = yolo.detect_image(image, crop=crop, count=count)
            r_image.save(dst_path)
            # r_image = yolo.detect_heatmap(image, dst_path)
    print("完成！")
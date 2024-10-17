import os
from tqdm import tqdm
from pathlib import Path

txtFolder = 'yolov8-obb/DOTA_dataset/labels/val'
imgFolder = 'yolov8-obb/DOTA_dataset/images/val'

def deleteImg(txtFolder, imgFolder):
    # 读取txt文件名
    txtName = set()
    for n in Path(txtFolder).rglob('*.txt'):
        txtName.add(Path(n).stem)
    
    # 遍历图片文件夹并删除不在txtName中的图片
    deleted_count = 0
    for m in tqdm(list(Path(imgFolder).rglob('*.jpg'))):
        if Path(m).stem not in txtName:
            os.remove(m)
            deleted_count += 1
    
    print(f'Finished! Deleted {deleted_count} images.')

if __name__ == '__main__':
    deleteImg(txtFolder, imgFolder)
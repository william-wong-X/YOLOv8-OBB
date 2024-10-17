import os
import random

from pathlib import Path
import numpy as np

from utils.utils import get_classes

# 类别文件
classes_path = 'model_data/dota_classes.txt'
# 训练集划分为9:1
train_percent = 0.9
# 数据集文件
dota_path = 'DOTA_dataset'
dota_sets = [('DOTA', 'train'), ('DOTA', 'val')]
classes, _ = get_classes(classes_path)

# 统计目标数量
photo_nums = np.zeros(len(dota_sets))
nums = np.zeros(len(classes))
def convert_annotation(data_root, chose, image_id, list_file):
    in_file = Path(data_root) / "labels" / chose / f"{image_id}.txt"
    annos = []
    with open(in_file, 'r') as f:
        for line in f:
            l = line.strip().split()
            annos.append(l)
    for anno in annos:
        list_file.write(" " + ",".join([str(a) for a in anno[1:]]) + "," + str(anno[0]))
        nums[int(anno[0])] = nums[int(anno[0])] + 1

if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(dota_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")
    
    print("Generate txt in ImageSets.")
    trainvalPath = Path(dota_path) / "labels" / "train"
    testPath     = Path(dota_path) / "labels" / "val"
    saveBasePath = Path(dota_path) / "ImageSets" / "Main"
    trainvalFile = trainvalPath.glob('*.txt')
    testFile     = testPath.glob('*txt')
    trainvalnum  = len(list(trainvalFile))
    trainnum     = int(trainvalnum*train_percent)
    testnum      = len(list(testFile))
    # 只能用一次
    trainvalFile = trainvalPath.glob('*.txt')
    testFile     = testPath.glob('*txt')
    
    list = range(trainvalnum)
    train = random.sample(list, trainnum)
    
    print("train and val size",trainvalnum)
    print("train size",trainnum)
    if not saveBasePath.exists():
        saveBasePath.mkdir(parents=True)
    ftrainval   = open(Path(saveBasePath) / 'trainval.txt', 'w')
    ftest       = open(Path(saveBasePath) / 'test.txt', 'w')
    ftrain      = open(Path(saveBasePath) / 'train.txt', 'w')
    fval        = open(Path(saveBasePath) / 'val.txt', 'w')
    
    # 将原本的train的数据分割为训练用的train和val
    name = [f"{f.stem}" for f in trainvalFile]
    for i in list:
        ftrainval.write(name[i] + '\n')  
        if i in train:  
            ftrain.write(name[i] + '\n')  
        else:  
            fval.write(name[i] + '\n')
    
    # 将原本的val当作test来使用        
    for file in testFile:
        ftest.write(f"{file.stem}" + "\n")
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Generate txt in ImageSets done.")
    
    print("Generate DOTA_train.txt and DOTA_val.txt for train.")
    type_index = 0
    for data_set, image_set in dota_sets:
        image_ids = open(Path(saveBasePath) / f"{image_set}.txt", 'r', encoding='utf-8').read().strip().split()
        list_file = open(Path(f"{data_set}_{image_set}.txt"), 'w', encoding='utf-8')
        for image_id in image_ids:
            list_file.write(f"{Path(dota_path).absolute() / 'images' / 'train' / f'{image_id}.jpg'}")
            convert_annotation(dota_path, 'train', image_id, list_file)
            list_file.write('\n')
        photo_nums[type_index] = len(image_ids)
        type_index += 1
        list_file.close()
    print("Generate DOTA_train.txt and DOTA_val.txt for train done.")
    
    def printTable(List1, List2):
        for i in range(len(List1[0])):
            print("|", end=' ')
            for j in range(len(List1)):
                print(List1[j][i].rjust(int(List2[j])), end=' ')
                print("|", end=' ')
            print()
            
    str_nums = [str(int(x)) for x in nums]
    tableData = [
        classes, str_nums
    ]
    colWidths = [0]*len(tableData)
    len1 = 0
    for i in range(len(tableData)):
        for j in range(len(tableData[i])):
            if len(tableData[i][j]) > colWidths[i]:
                colWidths[i] = len(tableData[i][j])
    printTable(tableData, colWidths)
    
    if photo_nums[0] <= 500:
        print("训练集数量小于500，属于较小的数据量，请注意设置较大的训练世代（Epoch）以满足足够的梯度下降次数（Step）。")
        
    if np.sum(nums) == 0:
        print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
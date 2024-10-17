# YOLOv8-OBB: 旋转框目标检测在Pytorch中的实现
---
## 训练步骤
### a、准备DOTA数据集
1、DOTA数据集存放在DOTA_dateset文件夹下
```bash
DOTA_dataset
  ├─images
  │  ├─test
  │  ├─train
  │  └─val
  └─labels
      ├─test
      ├─train
      └─val  
```
  
2、DOTA数据集的标注格式为  
30,464,64,440,84,468,50,492,0  
x0,y0,x1,y1,x2,y2,x3,y3,class  

3、对DOTA数据集进行分割
由于DOTA数据集是遥感数据集，所以其图片大小相对较大，无法直接用于训练模型，所以先使用[split_dota.py](split_dota.py)进行数据集的分割，但由于分割后的数据集存在空图像（没有目标的图像），所以要使用[rm_dota.py](rm_dota.py)删除空图像。  

分割后的DOTA数据集存放在DOTA_dateset_split文件夹下
```bash
DOTA_dataset_split
  ├─images
  │  ├─test
  │  ├─train
  │  └─val
  └─labels
      ├─test
      ├─train
      └─val  
```
### b、生成训练用数据集文件
1、运行[dota_annotation.py](dota_annotation.py)生成DOTA_train.txt和DOTA_val.txt。
### c、进行训练
1、对[train.py](train.py)文件进行修改，按照注释修改对应的路径。  

2、运行[train.py](train.py)进行训练，权值会生成在logs文件夹中。
### d、对训练好的模型进行性能评估
1、对[get_map.py](get_map.py)文件进行修改，按照注释修改对应的路径。  

2、运行[get_map.py](get_map.py)进行性能评估。  
| mAP | AP | Precision | Recall | F1 |  
| :---: | :---: | :---: | :---: | :---: |  
| 平均精度均值 | 精度均值 | 准确率 | 召回率 | F1分数 |  
### e、进行目标检测
1、目标检测需要用到两个文件，分别是[yolo.py](yolo.py)和[predict.py](predict.py)。需要按照注释修改yolo.py中的对应的路径。[predictfor.py](predictfor.py)用于批量检测。  

2、运行[predict.py](predict.py)，输入对应的图片路径进行检测。

### f、可视化界面
[qtgui.py](qtgui.py)是使用PyQt5编写的一个可视化界面。

## Reference
https://github.com/bubbliiiing/yolov8-pytorch   
https://github.com/ultralytics/ultralytics
import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image

from yolo import YOLO

class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Object Detection GUI')
        self.setFixedSize(1500, 800)  # 设置窗口固定大小
        
        # Layouts
        self.main_layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()
        
        # Labels to display images with fixed size
        self.original_image_label = QLabel('Original Image')
        self.original_image_label.setFixedSize(640, 640)  # 设置图像显示区域大小
        self.detected_image_label = QLabel('Detected Image')
        self.detected_image_label.setFixedSize(640, 640)  # 设置图像显示区域大小
        
        # Buttons
        self.load_button = QPushButton('选择图片')
        self.detect_button = QPushButton('检测')
        self.select_weights_button = QPushButton('切换权重')
        
        # Add widgets to layouts
        self.left_layout.addWidget(self.original_image_label)
        self.left_layout.addWidget(self.load_button)
        self.left_layout.addWidget(self.select_weights_button)
        self.right_layout.addWidget(self.detected_image_label)
        self.right_layout.addWidget(self.detect_button)
        
        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addLayout(self.right_layout)
        
        self.setLayout(self.main_layout)
        
        # Connect buttons to functions
        self.load_button.clicked.connect(self.load_image)
        self.detect_button.clicked.connect(self.detect_objects)
        self.select_weights_button.clicked.connect(self.select_weights)
        
        self.original_image = None
        self.image_path = None
        self.detected_image = None
        self.weights_path = None
        self.config_path = None
        
    def load_image(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if fileName:
            self.image_path = fileName
            self.original_image = cv2.imread(fileName)
            self.display_image(self.original_image, self.original_image_label)
    
    def select_weights(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select YOLO Weights File", "", "Weights Files (*.pth);;All Files (*)", options=options)
        if fileName:
            self.weights_path = fileName
            print(f"Selected weights file: {self.weights_path}")
    
    def detect_objects(self):
        if self.original_image is not None:
            self.detected_image = self.yolo_object_detection(self.original_image)
            self.display_image(self.detected_image, self.detected_image_label)
            self.save_detected_image(self.detected_image, self.image_path)
    
    def display_image(self, img, label):
        # Resize the image to fit the label
        img = cv2.resize(img, (label.width(), label.height()))
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        
        img = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        img = img.rgbSwapped()
        label.setPixmap(QPixmap.fromImage(img))
        
    def yolo_object_detection(self, image):
        yolo = YOLO(weights_path=self.weights_path)
        
        crop = False    # crop 指定了是否在单张图片预测后对目标进行截取
        count = False   # count 指定了是否进行目标的计数
        # 定义分割后的小块大小
        block_size = 640

        # 获取大图片的高度和宽度
        height, width, _ = image.shape

        # 初始化拼接后的图片
        output_image = np.zeros_like(image)

        # 分割并检测每个小块
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                # 计算当前小块的右下角坐标
                right = min(x + block_size, width)
                bottom = min(y + block_size, height)

                # 打印调试信息
                print(f"Processing block: x={x}, y={y}, right={right}, bottom={bottom}")

                # 提取当前小块
                block = image[y:bottom, x:right]

                # 检查提取的小块是否为空
                if block.size == 0:
                    print(f"Skipping empty block: x={x}, y={y}, right={right}, bottom={bottom}")
                    continue

                # 转换为PIL格式以适应YOLO检测
                block_pil = Image.fromarray(cv2.cvtColor(block, cv2.COLOR_BGR2RGB))

                # 检测
                r_image_pil = yolo.detect_image(block_pil, crop=crop, count=count)

                # 转换回OpenCV格式
                r_image = cv2.cvtColor(np.array(r_image_pil), cv2.COLOR_RGB2BGR)

                # 将检测结果拷贝回原图
                output_image[y:bottom, x:right] = r_image
            
        return output_image
    
    def save_detected_image(self, img, original_path):
        directory, filename = os.path.split(original_path)
        file_base, file_ext = os.path.splitext(filename)
        new_filename = f"{file_base}_detection{file_ext}"
        new_filepath = os.path.join(directory, new_filename)
        cv2.imwrite(new_filepath, img)
        print(f"Detected image saved as: {new_filepath}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ObjectDetectionApp()
    ex.show()
    sys.exit(app.exec_())

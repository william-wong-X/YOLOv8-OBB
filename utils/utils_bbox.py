import numpy as np
import torch
from torchvision.ops import nms
from mmcv.ops import nms_rotated
import pkg_resources as pkg
from utils.tal import dist2rbox

class DecodeBox():
    def __init__(self, num_classes, input_shape):
        super(DecodeBox, self).__init__()
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        
    def decode_box(self, inputs):
        # dbox  batch_size, 4, 8400
        # cls   batch_size, num_classes, 8400
        # angle batch_size, 1, 8400
        dbox, cls, origin_cls, angle, anchors, strides = inputs
        # 获得xywhr坐标
        dbox = dist2rbox(dbox, angle, anchors.unsqueeze(0), dim=1) * strides
        y = torch.cat((dbox, angle, cls.sigmoid()), 1).permute(0, 2, 1)
        # 进行归一化，到0~1之间
        y[:, :, :4] = y[:, :, :4] / torch.Tensor([self.input_shape[1], self.input_shape[0], self.input_shape[1], self.input_shape[0]]).to(y.device)
        return y
    
    def yolo_correct_boxes(self, out_put, input_shape, image_shape, letterbox_image):
        # 把y轴放前面是因为方便预测框和图像的宽高进行相乘
        box_xy = out_put[..., [0,1]]
        box_wh = out_put[..., [2,3]]
        angle = out_put[..., [4]]
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)
        
        if letterbox_image:
            # 这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            # new_shape指的是宽高缩放情况
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale
        
        box_xy = box_yx[..., ::-1]
        box_wh = box_hw[..., ::-1]
        rboxes = np.concatenate([box_xy, box_wh, angle], axis=-1)
        rboxes[:, [0, 2]] *= image_shape[1]
        rboxes[:, [1, 3]] *= image_shape[0]
        return rboxes
    
    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        # prediction [batch_size, num_anchors, 6+classes_num]
        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            '''
            对种类预测部分取max。
            class_conf  [num_anchors, 1]    种类置信度
            class_pred  [num_anchors, 1]    种类
            '''
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

            # 利用置信度进行第一轮筛选
            conf_mask = (class_conf[:, 0] >= conf_thres).squeeze()
            
            # 根据置信度进行预测结果的筛选
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.size(0):
                continue
            # detections  [num_anchors, 7]
            # 7的内容为：x, y, w, h, r, class_conf, class_pred
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

            # 获得预测结果中包含的所有种类
            unique_labels = detections[:, -1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()
                
            for c in unique_labels:
                # 获得某一类得分筛选后全部的预测结果
                detections_class = detections[detections[:, -1] == c]
                # 使用官方自带的非极大抑制会速度更快一些！
                # 筛选出一定区域内，属于同一种类得分最大的框
                dets, keep = nms_rotated(
                    detections_class[:, :5],
                    detections_class[:, 5],
                    nms_thres
                )
                max_detections = detections_class[keep]
                
                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
            
            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                output[i][:, :5] = self.yolo_correct_boxes(output[i], input_shape, image_shape, letterbox_image)
        return output
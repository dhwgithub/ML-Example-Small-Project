import cv2
import numpy as np
# import tensorflow as tf
from PIL import Image

import utils
import setting

"""
测试真实框在图像中的情况（画框显示）
"""
image_path = r"D:\Fils\CUR_WORK\Faster_RNN\Faster R-CNN\demo/1.jpg"
label_path = r"D:\Fils\CUR_WORK\Faster_RNN\Faster R-CNN\demo/1.txt"

gt_boxes = utils.load_gt_boxes(label_path)  # 把 ground truth boxes 的坐标读取出来
raw_image = cv2.imread(image_path)  # 将图片读取出来 (高，宽，通道数)
#
# image_with_gt_boxes = np.copy(raw_image)  # 复制原始图片
# utils.plot_boxes_on_image(image_with_gt_boxes, gt_boxes)  # 将 ground truth boxes 画在图片上
# Image.fromarray(image_with_gt_boxes).show()  # 展示画了 ground truth boxes 的图片

"""
锚框预测与展示
"""
target_scores = np.zeros(shape=[setting.grid_h, setting.grid_w, 9, 2])  # 下标0: background, 1: foreground（相当于于置信度）
target_bboxes = np.zeros(shape=[setting.grid_h, setting.grid_w, 9, 4])  # t_x, t_y, t_w, t_h （锚框与真实框之间的差距）
target_masks = np.zeros(shape=[setting.grid_h, setting.grid_w, 9])  # negative_samples: -1, positive_samples: 1

# 将 feature map 分成 45*60 个小块（特征图）
encoded_image = np.copy(raw_image)  # 再复制原始图片
for i in range(setting.grid_h):  # 45
    for j in range(setting.grid_w):  # 60
        for k in range(9):
            center_x = j * setting.grid_width + setting.grid_width * 0.5  # 计算此小块（映射到原图像）的中心点横坐标
            center_y = i * setting.grid_height + setting.grid_height * 0.5  # 计算此小块的中心点纵坐标

            xmin = center_x - setting.wandhG[k][0] * 0.5  # wandhG 是预测框的宽度和长度，xmin 是预测框在图上的左上角的横坐标
            ymin = center_y - setting.wandhG[k][1] * 0.5  # ymin 是预测框在图上的左上角的纵坐标
            xmax = center_x + setting.wandhG[k][0] * 0.5  # xmax 是预测框在图上的右下角的纵坐标
            ymax = center_y + setting.wandhG[k][1] * 0.5  # ymax 是预测框在图上的右下角的纵坐标

            # ignore cross-boundary anchors
            if (xmin > -5) & (ymin > -5) & (xmax < (setting.image_width + 5)) & (ymax < (setting.image_height + 5)):
                anchor_boxes = np.array([xmin, ymin, xmax, ymax])  # size = 1
                anchor_boxes = np.expand_dims(anchor_boxes, axis=0)

                # compute iou between this anchor and all ground-truth boxes in image.
                ious = utils.compute_iou(anchor_boxes, gt_boxes)
                positive_masks = ious > setting.pos_thresh
                negative_masks = ious < setting.neg_thresh

                if np.any(positive_masks):  # 如果有一个为 True，则返回 True，即存在正样例则进入
                    utils.plot_boxes_on_image(encoded_image, anchor_boxes, thickness=1)
                    # print("=> Encoding positive sample: %d, %d, %d" % (i, j, k))
                    cv2.circle(encoded_image,
                               center=(int(0.5 * (xmin + xmax)), int(0.5 * (ymin + ymax))),
                               radius=1,
                               color=[255, 0, 0],
                               thickness=4)  # 正预测框的中心点用红圆表示

                    target_scores[i, j, k, 1] = 1.  # 表示检测到物体，由于确定该位置一定有对象，因此置信度是1
                    target_masks[i, j, k] = 1  # labeled as a positive sample

                    # find out which ground-truth box matches this anchor
                    max_iou_idx = np.argmax(ious)  # 找到最大IOU的下标
                    selected_gt_boxes = gt_boxes[max_iou_idx]  # 找到与当前锚框最大IOU的真实框
                    target_bboxes[i, j, k] = utils.compute_regression(selected_gt_boxes, anchor_boxes[0])

                if np.all(negative_masks):  # 且操作，全部为真才为真
                    target_scores[i, j, k, 0] = 1.  # 表示是背景
                    target_masks[i, j, k] = -1  # labeled as a negative sample
                    cv2.circle(encoded_image,
                               center=(int(0.5 * (xmin + xmax)), int(0.5 * (ymin + ymax))),
                               radius=1,
                               color=[0, 0, 0],
                               thickness=4)  # 负预测框的中心点用黑圆表示

# Image.fromarray(encoded_image).show()

"""
绘制回归框
"""
faster_decode_image = np.copy(raw_image)

pred_bboxes = np.expand_dims(target_bboxes, 0).astype(np.float32)
pred_scores = np.expand_dims(target_scores, 0).astype(np.float32)
# print(pred_bboxes.shape, pred_scores.shape)  # (1, 45, 60, 9, 4) (1, 45, 60, 9, 2)

pred_scores, pred_bboxes = utils.decode_output(pred_bboxes, pred_scores)

utils.plot_boxes_on_image(faster_decode_image, pred_bboxes, color=[255, 0, 0])  # red boundig box
Image.fromarray(np.uint8(faster_decode_image)).show()

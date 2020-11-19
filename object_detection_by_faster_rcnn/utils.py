# coding: utf-8

import random
import os
import cv2
import numpy as np
import tensorflow as tf

import setting


def load_gt_boxes(label_path):
    """
    返回真实框坐标值，[-1, 4].
    xmin, ymin, xmax, ymax
    """
    bbs = open(label_path).readlines()[1:]  # 排除第一行的：% bbGt version=3
    roi = np.zeros([len(bbs), 4])
    for iter_, bb in zip(range(len(bbs)), bbs):
        bb = bb.replace('\n', '').split(' ')

        # bbtype = bb[0]  # 对象类型，暂无用到
        bba = np.array([float(bb[i]) for i in range(1, 5)])  # 2个坐标+2个参数（宽+高）

        # 后面信息无说明
        # occ = float(bb[5])
        # bbv = np.array([float(bb[i]) for i in range(6, 10)])

        # 似乎是排除其他影响，如只检测人
        # ignore = int(bb[10])
        # ignore = ignore or (bbtype != 'person')
        # ignore = ignore or (bba[3] < 40)

        # 对于右下角坐标需要用宽高加上相对左上角的坐标
        bba[2] += bba[0]
        bba[3] += bba[1]

        roi[iter_, :4] = bba
    return roi


def plot_boxes_on_image(show_image_with_boxes, boxes, color=[0, 0, 255], thickness=2):
    """
    为给定图像画上检测框（传入信息）
    :param show_image_with_boxes: 传入的BGR图像
    :param boxes: 检测框信息
    :param color:
    :param thickness:
    :return:
    """
    for box in boxes:
        cv2.rectangle(show_image_with_boxes,
                      pt1=(int(box[0]), int(box[1])),
                      pt2=(int(box[2]), int(box[3])),
                      color=color,
                      thickness=thickness)
    show_image_with_boxes = cv2.cvtColor(show_image_with_boxes, cv2.COLOR_BGR2RGB)
    return show_image_with_boxes


def compute_iou(boxes1, boxes2):
    """
    锚框集合、gt框集合
    (xmin, ymin, xmax, ymax)
    boxes1 shape:  [-1, 4], boxes2 shape: [-1, 4]
    """
    # np.maximum(X, Y, out=None)
    # X和Y逐位进行比较,选择最大值，最少接受两个参数。numpy.minimum()同理
    # >>> np.maximum([2, 3, 4], [1, 5, 2])
    # array([2, 5, 4])
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2], )  # left_up=[xmin2, ymin2]
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])  # right_down=[xmax1, ymax1]

    inter_wh = np.maximum(right_down - left_up, 0.0)  # 差最小为0（即不存在交集），也不会影响后面求面积
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]  # 交集面积

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union_area = boxes1_area + boxes2_area - inter_area

    return inter_area / union_area


def compute_regression(box1, box2):
    """将锚框向真实框回归，存在偏移和缩放
    返回水平偏移/缩放、垂直偏移/缩放量
    box1: ground-truth boxes
    box2: anchor boxes
    """
    target_reg = np.zeros(shape=[4, ])
    # 真实框的宽高
    w1 = box1[2] - box1[0]
    h1 = box1[3] - box1[1]
    # 锚框的宽高
    w2 = box2[2] - box2[0]
    h2 = box2[3] - box2[1]

    target_reg[0] = (box1[0] - box2[0]) / w2  # 水平偏移量
    target_reg[1] = (box1[1] - box2[1]) / h2  # 垂直偏移量
    target_reg[2] = np.log(w1 / w2)  # 水平缩放量
    target_reg[3] = np.log(h1 / h2)  # 垂直缩放量

    return target_reg


def decode_output(pred_bboxes, pred_scores, score_thresh=0.5):
    """返回检测框信息
    pred_bboxes shape: [1, grid_h, grid_w, 9, 4] 表示一共 grid_h*grid_w*9 个预测框，每个预测框都包含着两个平移量和两个尺度因子
    pred_scores shape: [1, grid_h, grid_w, 9, 2] 表示在 grid_h*grid_w*9 个预测框中:
                        [1, i, j, k, 0] 表示第 i 行第 j 列中的第 k 个预测框中包含的是背景的概率
                        [1, i, j, k, 1] 表示第 i 行第 j 列中的第 k 个预测框中包含的是检测物体的概率
    """
    grid_x, grid_y = tf.range(setting.grid_w, dtype=tf.int32), tf.range(setting.grid_h, dtype=tf.int32)
    # tf.meshgrid(x, y) 最后两个矩阵的维数相同，用b的维数作为行，用a的维数作为列
    # a=[0,5,10]
    # b=[0,5,15,20,25]
    # A,B=tf.meshgrid(a,b)
    # with tf.Session() as sess:
    #   print (A.eval())
    #   print (B.eval())
    #
    # Output:
    # -------------
    # [[ 0  5 10]
    #  [ 0  5 10]
    #  [ 0  5 10]
    #  [ 0  5 10]
    #  [ 0  5 10]]
    # [[ 0  0  0]
    #  [ 5  5  5]
    #  [15 15 15]
    #  [20 20 20]
    #  [25 25 25]]
    # -------------
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)  # (45, 60) 特征图（二维）矩阵表示
    grid_x, grid_y = tf.expand_dims(grid_x, -1), tf.expand_dims(grid_y, -1)  # (45, 60, 1) 负数扩维，表示从右扩充维度
    grid_xy = tf.stack([grid_x, grid_y], axis=-1)  # (45, 60, 1, 2) 包含着所有特征图中小块的左上角的坐标
    '''grid_xy
    tf.Tensor(
    [[[[0 0]]
      [[1 0]]
      [[2 0]]
      [[3 0]]]
      ...

     [[[0 1]]
      [[1 1]]
      [[2 1]]
      [[3 1]]]
      ...

     [[[0 2]]
      [[1 2]]
      [[2 2]]
      [[3 2]]]]..., shape=(3, 4, 1, 2), dtype=int32)
    '''
    center_xy = grid_xy * 16 + 8  # 计算原始图像上每个小块的中心 center_xy（特征图中一个小块能表示原始图像中一块 16*16 的区域）
    center_xy = tf.cast(center_xy, tf.float32)
    anchor_xymin = center_xy - 0.5 * setting.wandhG  # 计算预测框的左上角坐标（每个特征图块减去提前规定好的锚框的一半）

    # 具体公式操作参考compute_regression的计算方式
    xy_min = pred_bboxes[..., 0:2] * setting.wandhG[:, 0:2] + anchor_xymin  # 回归框的左上角坐标（compute_regression()）
    xy_max = tf.exp(pred_bboxes[..., 2:4]) * setting.wandhG[:, 0:2] + xy_min  # 回归框的右下角坐标（compute_regression()）

    pred_bboxes = tf.concat([xy_min, xy_max], axis=-1)  # （1， 45， 60， 9） 包含着回归框左上角坐标和右下角坐标
    pred_scores = pred_scores[..., 1]  # （1， 45， 60） 指的是每个框中含有检测目标的概率（称为得分）

    score_mask = pred_scores > score_thresh  # （1， 45， 60， 9）
    pred_bboxes = tf.reshape(pred_bboxes[score_mask], shape=[-1, 4]).numpy()  # （86， 4） 每个检测框的左上角和右下角的坐标
    pred_scores = tf.reshape(pred_scores[score_mask], shape=[-1, ]).numpy()  # （86，） 每个检测框中的内容是检测物的概率

    return pred_scores, pred_bboxes


def nms(pred_boxes, pred_score, iou_thresh=0.1):
    """去除掉那些重叠率较高但得分较低的预测框
    其流程为：
        取出所有预测框中得分最高的一个，并将这个预测框跟其他的预测框进行 IOU 计算；
        将 IOU 值大于 0.1 的预测框视为与刚取出的得分最高的预测框表示了同一个检测物，故去掉；
        重复以上操作，直到所有其他的预测框都被去掉为止。
    pred_boxes shape: [-1, 4]
    pred_score shape: [-1,]
    """
    selected_boxes = []
    while len(pred_boxes) > 0:
        max_idx = np.argmax(pred_score)
        selected_box = pred_boxes[max_idx]
        selected_boxes.append(selected_box)

        pred_boxes = np.concatenate([pred_boxes[:max_idx], pred_boxes[max_idx+1:]])
        pred_score = np.concatenate([pred_score[:max_idx], pred_score[max_idx+1:]])
        ious = compute_iou(selected_box, pred_boxes)

        iou_mask = ious <= iou_thresh
        pred_boxes = pred_boxes[iou_mask]
        pred_score = pred_score[iou_mask]

    return np.array(selected_boxes)


def encode_label(gt_boxes):
    """
    对于每个真实框计算其在特征图中的信息
    每个预测框的得分和训练变量（与demo一样）
    :param gt_boxes:
    :return:
        target_scores：目标得分，即判断一张图片中所有检测框中是背景的概率和是检测物的概率，其形状为 (1, 45, 60, 9, 2)
        target_bboxes：目标检测框，即一张图片中所有检测框用于回归的训练变量，其形状为 (1, 45, 60, 9, 4)
        target_masks：目标掩膜，其值包括 -1，0，1。
            -1 表示这个检测框中是背景
            1 表示这个检测框中是检测物
            0 表示这个检测框中既不是背景也不是检测物
    """
    target_scores = np.zeros(shape=[setting.grid_h, setting.grid_w, 9, 2])  # 0: background, 1: foreground, ,
    target_bboxes = np.zeros(shape=[setting.grid_h, setting.grid_w, 9, 4])  # t_x, t_y, t_w, t_h
    target_masks = np.zeros(shape=[setting.grid_h, setting.grid_w, 9])  # negative_samples: -1, positive_samples: 1

    for i in range(setting.grid_h):  # y: height
        for j in range(setting.grid_w):  # x: width
            for k in range(9):
                center_x = j * setting.grid_width + setting.grid_width * 0.5
                center_y = i * setting.grid_height + setting.grid_height * 0.5

                xmin = center_x - setting.wandhG[k][0] * 0.5
                ymin = center_y - setting.wandhG[k][1] * 0.5
                xmax = center_x + setting.wandhG[k][0] * 0.5
                ymax = center_y + setting.wandhG[k][1] * 0.5
                # print(xmin, ymin, xmax, ymax)

                # ignore cross-boundary anchors
                if (xmin > -5) & (ymin > -5) & (xmax < (setting.image_width+5)) & (ymax < (setting.image_height+5)):
                    anchor_boxes = np.array([xmin, ymin, xmax, ymax])
                    anchor_boxes = np.expand_dims(anchor_boxes, axis=0)

                    # compute iou between this anchor and all ground-truth boxes in image.
                    ious = compute_iou(anchor_boxes, gt_boxes)
                    positive_masks = ious >= setting.pos_thresh
                    negative_masks = ious <= setting.neg_thresh

                    if np.any(positive_masks):
                        target_scores[i, j, k, 1] = 1.
                        target_masks[i, j, k] = 1  # labeled as a positive sample

                        # find out which ground-truth box matches this anchor
                        max_iou_idx = np.argmax(ious)
                        selected_gt_boxes = gt_boxes[max_iou_idx]
                        target_bboxes[i, j, k] = compute_regression(selected_gt_boxes, anchor_boxes[0])

                    if np.all(negative_masks):
                        target_scores[i, j, k, 0] = 1.
                        target_masks[i, j, k] = -1  # labeled as a negative sample

    return target_scores, target_bboxes, target_masks


def process_image_label(image_path, label_path):
    """
    数据预处理
    :param image_path:
    :param label_path:
    :return:
    """
    raw_image = cv2.imread(image_path)
    gt_boxes = load_gt_boxes(label_path)
    target = encode_label(gt_boxes)
    return raw_image / 255., target


def create_image_label_path_generator(synthetic_dataset_path):
    """
    加载数据
    :param synthetic_dataset_path:
    :return:
    """
    image_num = setting.datasets_num
    image_label_paths = [(os.path.join(synthetic_dataset_path, "image/%s.jpg" % id.split('.')[0]),
                          os.path.join(synthetic_dataset_path, "imageAno/%s.txt" % id.split('.')[0]))
                         for id in os.listdir(os.path.join(synthetic_dataset_path, "image"))]

    while True:
        random.shuffle(image_label_paths)
        for i in range(image_num):
            yield image_label_paths[i]


def DataGenerator(synthetic_dataset_path, batch_size):
    """
    数据生成器
    """
    image_label_path_generator = create_image_label_path_generator(synthetic_dataset_path)
    while True:
        images = np.zeros(shape=[batch_size, setting.image_height, setting.image_width, 3],
                          dtype=np.float)
        target_scores = np.zeros(shape=[batch_size, setting.grid_h, setting.grid_w, 9, 2],
                                 dtype=np.float)
        target_bboxes = np.zeros(shape=[batch_size, setting.grid_h, setting.grid_w, 9, 4],
                                 dtype=np.float)
        target_masks = np.zeros(shape=[batch_size, setting.grid_h, setting.grid_w, 9],
                                dtype=np.int)

        for i in range(batch_size):
            image_path, label_path = next(image_label_path_generator)
            image, target = process_image_label(image_path, label_path)

            images[i] = image
            target_scores[i] = target[0]
            target_bboxes[i] = target[1]
            target_masks[i] = target[2]

        yield images, target_scores, target_bboxes, target_masks


def compute_loss(target_scores, target_bboxes, target_masks, pred_scores, pred_bboxes):
    """
    计算损失
    target_scores shape: [1, 45, 60, 9, 2],  pred_scores shape: [1, 45, 60, 9, 2]
    target_bboxes shape: [1, 45, 60, 9, 4],  pred_bboxes shape: [1, 45, 60, 9, 4]
    target_masks  shape: [1, 45, 60, 9]
    """
    score_loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_scores,
                                                         logits=pred_scores)
    # 只关心前景的得分损失
    foreground_background_mask = (np.abs(target_masks) == 1).astype(np.int)
    score_loss = tf.reduce_sum(score_loss * foreground_background_mask, axis=[1, 2, 3]) \
                 / np.sum(foreground_background_mask)
    score_loss = tf.reduce_mean(score_loss)

    boxes_loss = tf.abs(target_bboxes - pred_bboxes)
    # soomth L1 损失
    boxes_loss = 0.5 * tf.pow(boxes_loss, 2) * tf.cast(boxes_loss < 1, tf.float32) \
                 + (boxes_loss - 0.5) * tf.cast(boxes_loss >= 1, tf.float32)
    boxes_loss = tf.reduce_sum(boxes_loss, axis=-1)
    # 只关心正预测框的回归损失
    foreground_mask = (target_masks > 0).astype(np.float32)
    boxes_loss = tf.reduce_sum(boxes_loss * foreground_mask, axis=[1, 2, 3]) / np.sum(foreground_mask)
    boxes_loss = tf.reduce_mean(boxes_loss)

    return score_loss, boxes_loss

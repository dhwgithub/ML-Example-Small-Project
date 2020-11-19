import numpy as np

# 训练参数
EPOCHS = 10
STEPS = 8000
batch_size = 1
lambda_scale = 1.
learn_rate = 1e-4
train_dataset_path = r"./datasets/train"
test_img_path = r"./datasets/test/image"
predict_output = r'./output'

# 数据集参数
datasets_num = 8000

image_height = 720
image_width = 960

grid_width = 16  # 网格的长宽都是16，因为从原始图片到 feature map 经历了16倍的缩放
grid_height = 16

# 因为得到的 feature map 的长宽都是原始图片的 1/16，所以这里 45=720/16，60=960/16。
# 表示一共 grid_h * grid_w * 9 个预测框
grid_h = 45
grid_w = 60

# 阈值
pos_thresh = 0.5
neg_thresh = 0.1
iou_thresh = 0.5

# 包含着 9 个预测/锚框的宽度和长度（这是经过 kmeans 算法计算过的结果）get_bboxes_by_kmeans.py
wandhG = np.array([[189., 206.],
                   [68., 172.],
                   [45., 160.],
                   [103., 243.],
                   [41., 109.],
                   [97.,  74.],
                   [126., 166.],
                   [91., 130.],
                   [155., 103.]], dtype=np.float32)

import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from rpn import RPNplus
import setting
import utils

if not os.path.exists(setting.predict_output):
    os.mkdir(setting.predict_output)

model = RPNplus()
fake_data = np.ones(shape=[1, setting.image_height, setting.image_width, 3]).astype(np.float32)
model(fake_data)  # initialize model to load weights
model.load_weights("./RPN.h5")

for img_name in os.listdir(setting.test_img_path):  # 预测200张
    image_path = os.path.join(setting.test_img_path, img_name)
    raw_image = cv2.imread(image_path)
    image_data = np.expand_dims(raw_image / 255., 0)

    pred_scores, pred_bboxes = model(image_data)
    pred_scores = tf.nn.softmax(pred_scores, axis=-1)

    pred_scores, pred_bboxes = utils.decode_output(pred_bboxes, pred_scores, 0.9)
    pred_bboxes = utils.nms(pred_bboxes, pred_scores, 0.5)

    utils.plot_boxes_on_image(raw_image, pred_bboxes)
    save_path = os.path.join(setting.predict_output, img_name)
    print("=> saving prediction results into %s" % save_path)
    Image.fromarray(raw_image).save(save_path)

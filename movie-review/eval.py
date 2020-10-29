# -*- coding: utf-8 -*-
"""
开发及验证模块
"""
import tensorflow as tf

import settings
import models
import dataset


# 数据集
x = tf.placeholder(tf.int32, [None, None])
y = tf.placeholder(tf.float32, [None, 1])

# emb层的dropout保留率
emb_keep = tf.placeholder(tf.float32)
# rnn层的dropout保留率
rnn_keep = tf.placeholder(tf.float32)

# 创建一个模型
model = models.Model(x, y, emb_keep, rnn_keep)

# 创建一个数据集对象
data = dataset.Dataset(1)  # 0-训练集 1-开发集 2-测试集

# 移动平均变量
restore_variables = model.ema.variables_to_restore()

# 使用移动平均变量进行覆盖
saver = tf.train.Saver(restore_variables)

with tf.Session() as sess:
    # 加载最新的模型
    ckpt = tf.train.get_checkpoint_state(settings.CKPT_PATH)
    saver.restore(sess, ckpt.model_checkpoint_path)

    # 计算并输出acc
    acc = sess.run([model.acc],
                   {model.data: data.data,
                    model.label: data.labels,
                    model.emb_keep: 1.0,
                    model.rnn_keep: 1.0})
    print('acc is ', acc)

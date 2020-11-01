# -*- coding: utf-8 -*-
import tensorflow as tf
from dataset import tokenizer
import gc

import settings
from log import my_log as logging

# 设置按需分配内存
logging.info('========== 清空缓存 ==========')
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# 每次构建模型前清空之前模型占用的内存
tf.keras.backend.clear_session()
# 回收没有被使用的空间
gc.collect()

# 构建模型
logging.info('========== 构建模型 ==========')
model = tf.keras.Sequential([
    # 不定长度的输入
    tf.keras.layers.Input((None, )),

    # 词嵌入层
    tf.keras.layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=settings.HIDDEN_NUM),

    # 第一个LSTM层，返回序列作为下一层的输入
    tf.keras.layers.LSTM(settings.HIDDEN_NUM, dropout=0.5, return_sequences=True),

    # 第二个LSTM层，返回序列作为下一层的输入
    tf.keras.layers.LSTM(settings.HIDDEN_NUM, dropout=0.5, return_sequences=True),

    # 对每一个时间点的输出都做softmax，预测下一个词的概率
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(tokenizer.vocab_size,
                                                          activation='softmax')),
])

# 查看模型结构
model.summary()

# 配置优化器和损失函数
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.categorical_crossentropy)

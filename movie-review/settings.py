# -*- coding: utf-8 -*-
"""
全局参数设置
"""

# 源数据路径
ORIGIN_NEG = r'datasets/rt-polarity.neg'
ORIGIN_POS = r'datasets/rt-polarity.pos'

# 解码后源数据路径
NEG_TXT = r'datasets/neg.txt'
POS_TXT = r'datasets/pos.txt'

# 词汇表
VOCAB_SIZE = 10000
VOCAB_PATH = r'datasets/vocab.txt'

# 词向量路径
NEG_VEC = r'datasets/neg.vec'
POS_VEC = r'datasets/pos.vec'

# 数据集占比及路径
TRAIN_RATE = 0.8
DEV_RATE = 0.1
TEST_RATE = 0.1

TRAIN_DATA = r'datasets/train'
DEV_DATA = r'datasets/dev'
TEST_DATA = r'datasets/test'

# 网络结构设置
HIDDEN_SIZE = 128
NUM_LAYERS = 2

EMA_RATE = 0.99  # 移动平均衰减率
LEARN_RATE = 0.0001  # 初始学习率
LR_DECAY = 0.99  # 学习率
LR_DECAY_STEP = 1000  # 学习率衰减频率

# 训练参数
TRAIN_TIMES = 2000
BATCH_SIZE = 64

EMB_KEEP_PROB = 0.5  # emb层dropout保留率
RNN_KEEP_PROB = 0.5  # rnn层dropout保留率

SHOW_STEP = 10  # 输出loss频率
SAVE_STEP = 100  # 保存模型频率

CKPT_PATH = 'ckpt'  # 模型保存路径
MODEL_NAME = 'model'  # 模型名称

# -*- coding: utf-8 -*-
"""
全局变量
"""
# Log
LOG_PATH = r'logs/poetry.logs'
LOG_BASIC_FORMAT = "%(asctime)s %(levelname)s: %(message)s"
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# 数据集路径
DATASET_PATH = r'datasets/poetry.txt'

# 最佳权重保存路径
BEST_MODEL_PATH = r'datasets/best_model.h5'

# 禁用词，包含如下字符的唐诗将被忽略
DISALLOWED_WORDS = ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']']

# 诗句最大长度
MAX_LEN = 64

# 最小词频（用于过滤低频词）
MIN_WORD_FREQUENCY = 8

# 每个epoch训练完成后，随机生成SHOW_NUM首古诗作为展示
SHOW_NUM = 3

# 训练
TRAIN_EPOCHS = 20
BATCH_SIZE = 4

# LSTM每层神经元个数
HIDDEN_NUM = 128

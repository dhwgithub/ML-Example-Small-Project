# -*- coding: utf-8 -*-
"""
检验最终效果
"""
import tensorflow as tf

from dataset import tokenizer
import settings
import utils
from log import my_log as logging

# 加载训练好的模型
logging.info('start test ...')
model = tf.keras.models.load_model(settings.BEST_MODEL_PATH)

# 随机生成一首诗
logging.info('random create poetry: %s', utils.generate_random_poetry(tokenizer, model))

# 给出部分信息的情况下，随机生成剩余部分
logging.info('have_pre create poetry: %s', utils.generate_random_poetry(tokenizer,
                                                                        model,
                                                                        s='床前明月光，'))

# 生成藏头诗
logging.info('create acrostic poetry: %s', utils.generate_acrostic(tokenizer,
                                                                   model,
                                                                   head='计算视觉'))

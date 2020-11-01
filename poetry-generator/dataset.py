# -*- coding: utf-8 -*-
"""
数据处理模块

源数据格式：
    题目：诗
"""
from collections import Counter
import math
import numpy as np
import tensorflow as tf

import settings
from log import my_log as logging


class Tokenizer:
    """
    词编码器
    """
    def __init__(self, token_dict):
        logging.info('get dict ...')

        # 词 -> 编号的映射
        self.token_dict = token_dict

        # 编号->词的映射
        self.token_dict_rev = {value: key for key, value in self.token_dict.items()}

        # 词汇表大小
        self.vocab_size = len(self.token_dict)

    def id_to_token(self, token_id):
        """
        根据编码查token
        :param token_id:
        :return:
        """
        return self.token_dict_rev[token_id]

    def token_to_id(self, token):
        """
        给定token查编码
        若无则返回[UNK]的编码
        :param token:
        :return:
        """
        return self.token_dict.get(token, self.token_dict['[UNK]'])

    def encode(self, tokens):
        """
        加入起始、结束符
        :param tokens:
        :return:
        """
        # 加上开始标记
        token_ids = [self.token_to_id('[CLS]'), ]

        # 加入字符串编号序列
        for token in tokens:
            token_ids.append(self.token_to_id(token))

        # 加上结束标记
        token_ids.append(self.token_to_id('[SEP]'))

        return token_ids

    def decode(self, token_ids):
        """
        编码的逆过程，去除起始和终止符号
        :param token_ids:
        :return:
        """
        # 起止标记字符特殊处理
        spec_tokens = {'[CLS]', '[SEP]'}

        # 保存解码出的字符的list
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token(token_id)
            if token in spec_tokens:
                continue
            tokens.append(token)

        # 拼接字符串
        return ''.join(tokens)


class PoetryDataGenerator:
    """
    古诗数据集生成器
    """
    def __init__(self, data, random=False):
        # 数据集
        self.data = data

        # batch size
        self.batch_size = settings.BATCH_SIZE
        logging.info('batch_size: %s', settings.BATCH_SIZE)

        # 每个epoch迭代的步数
        self.steps = int(math.floor(len(self.data) / self.batch_size))
        logging.info('data num: %s', len(self.data))

        # 每个epoch开始时是否随机混洗
        self.random = random

    def sequence_padding(self, data, length=None, padding=None):
        """
        填充数据
        :param data:
        :param length: 填充后的长度，默认最长
        :param padding: 填充的数据编码，默认[PAD]的对应编码
        :return:
        """
        # 设置默认填充长度
        if length is None:
            length = max(map(len, data))

        # 设置默认填充编码
        if padding is None:
            padding = tokenizer.token_to_id('[PAD]')

        # 开始填充
        outputs = []
        for line in data:
            padding_length = length - len(line)
            if padding_length > 0:
                outputs.append(np.concatenate([line, [padding] * padding_length]))
            else:
                outputs.append(line[:length])

        return np.array(outputs)

    def __len__(self):
        return self.steps

    def __iter__(self):
        total = len(self.data)

        # 是否随机混洗
        if self.random:
            np.random.shuffle(self.data)

        # 迭代一个epoch，每次yield一个batch
        for start in range(0, total, self.batch_size):

            end = min(start + self.batch_size, total)
            batch_data = []

            # 逐一对古诗进行编码
            for single_data in self.data[start: end]:
                batch_data.append(tokenizer.encode(single_data))

            # 填充为相同长度
            batch_data = self.sequence_padding(batch_data)

            '''
            yield 中前面部分是数据 x,后面部分是标签 y。
            将诗的内容错开一位分别作为数据和标签，举个例子：
            假设有诗是“[CLS]床前明月光，疑是地上霜。举头望明月，低头思故乡。[SEP]”，
            
            则数据为“[CLS]床前明月光，疑是地上霜。举头望明月，低头思故乡。”；
              标签为“床前明月光，疑是地上霜。举头望明月，低头思故乡。[SEP]”。
            
            两者一一对应，y 是 x 中每个位置的下一个字符。

            以字符的形式举例是为了方便理解，实际上不论是数据还是标签，
            都是使用 tokenizer 编码后的编号序列。

            还有一点不同的是，标签部分使用了 one-hot 进行处理，而数据部分没有使用。
            原因在于，数据部分准备输入词嵌入层，而词嵌入层的输入不需要进行 one-hot；
            而标签部分，需要和模型的输出计算交叉熵，输出层的激活函数是 softmax，
            所以标签部分也要转成相应的 shape，故使用 one-hot 形式。
            '''
            yield batch_data[:, :-1], tf.one_hot(batch_data[:, 1:], tokenizer.vocab_size)
            del batch_data

    def for_fit(self):
        """
        创建用于训练的生成器
        :return:
        """
        # 死循环，当数据训练一个epoch之后，重新迭代数据
        while True:
            yield from self.__iter__()


# 加载数据集
logging.info('========== 读取数据集 ==========')
logging.info('dataset path: %s', settings.DATASET_PATH)
with open(settings.DATASET_PATH, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines = [line.replace('：', ':') for line in lines]

# 数据集列表
poetry = []
for line in lines:

    # 有且只能有一个冒号用来分割标题
    if line.count(':') != 1:
        continue

    # 后半部分不能包含禁止词
    __, last_part = line.split(':')
    ignore_flag = False
    for dis_word in settings.DISALLOWED_WORDS:
        if dis_word in last_part:
            ignore_flag = True
            break
    if ignore_flag:
        continue

    # 长度不能超过最大长度
    if len(last_part) > settings.MAX_LEN - 2:
        continue

    poetry.append(last_part.replace('\n', ''))

# 统计词频
counter = Counter()
for line in poetry:
    counter.update(line)

# 过滤掉低频词
_tokens = [(token, count)
           for token, count in counter.items()
           if count >= settings.MIN_WORD_FREQUENCY]

# 按词频排序
_tokens = sorted(_tokens, key=lambda x: -x[1])

# 去掉词频，只保留词列表
_tokens = [token for token, count in _tokens]

# 将特殊词和数据集中的词拼接起来
_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + _tokens

# 创建词典 token -> id 映射关系
token_id_dict = dict(zip(_tokens, range(len(_tokens))))

# 使用新词典重新建立分词器
tokenizer = Tokenizer(token_id_dict)

# 混洗数据
np.random.shuffle(poetry)

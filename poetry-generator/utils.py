# -*- coding: utf-8 -*-
"""
随机生成古诗词 和 生成藏头诗
"""
import logging
from logging import handlers
import numpy as np

import settings


def generate_random_poetry(tokenizer, model, s=''):
    """
    随机生成一首诗
    :param tokenizer: 分词器
    :param model: 用于生成古诗的模型
    :param s: 用于生成古诗的起始字符串，默认为空串
    :return: 一个字符串，表示一首古诗
    """
    # 将初始字符串转成token
    token_ids = tokenizer.encode(s)

    # 去掉结束标记[SEP]
    token_ids = token_ids[:-1]

    while len(token_ids) < settings.MAX_LEN:

        token_ids, target = create_token_ids(model, token_ids)

        # 若不存在则退出
        if target == 3:
            break

    # 返回解码后的信息（即汉字）
    return tokenizer.decode(token_ids)


def generate_acrostic(tokenizer, model, head):
    """
    随机生成一首藏头诗
    :param tokenizer: 分词器
    :param model: 用于生成古诗的模型
    :param head: 藏头诗的头
    :return: 一个字符串，表示一首古诗
    """
    # 使用空串初始化token_ids，加入[CLS]
    token_ids = tokenizer.encode('')
    token_ids = token_ids[:-1]

    # 标点符号，这里简单的只把逗号和句号作为标点
    punctuations = ['，', '。']
    punctuation_ids = {tokenizer.token_to_id(token) for token in punctuations}

    # 缓存生成的诗的list
    poetry = []

    # 对于藏头诗中的每一个字，都生成一个短句
    for ch in head:
        # 先记录下这个字
        poetry.append(ch)

        # 将藏头诗的字符转成token id
        token_id = tokenizer.token_to_id(ch)

        # 加入到列表中去
        token_ids.append(token_id)

        # 开始生成一个短句
        while True:

            token_ids, target = create_token_ids(model, token_ids)

            # 只有不是特殊字符时，才保存到poetry里面去
            if target > 3:
                poetry.append(tokenizer.id_to_token(target))
            if target in punctuation_ids:
                break

    return ''.join(poetry)


def create_token_ids(model, token_ids):
    """
    根据 模型预测 和 token_ids 返回处理后的 token_ids 和 预测值的下标
    :param model: 模型
    :param token_ids: 缺少预测值的 token_ids
    :return:
    """
    # 进行预测
    output = model(np.array([token_ids, ], dtype=np.int32))
    # 只保留第一个样例（我们输入的样例数只有1）的预测且不包含[PAD][UNK][CLS]的概率分布
    _probas = output.numpy()[0, -1, 3:]
    del output

    # 按照出现概率降序排列并取前100个（索引）
    p_args = _probas.argsort()[::-1][:100]
    # 排列后的概率顺序
    p = _probas[p_args]

    # 对概率归一
    p = p / sum(p)
    # 按照每个元素出现的概率，随机选择 len(p) - 1 中的一个数
    target_index = np.random.choice(len(p), p=p)
    # 将索引 + 3，因为之前排除了前3个特殊元素的索引
    target = p_args[target_index] + 3

    # 保存索引
    token_ids.append(target)

    return token_ids, target


def log_setting():
    """
    LOG设置
    :return:
    """
    # 日志总设置
    my_log = logging.getLogger()
    my_log.setLevel('DEBUG')

    # 日志格式设置
    formatter = logging.Formatter(settings.LOG_BASIC_FORMAT, settings.LOG_DATE_FORMAT)

    # 控制台输出设置（不打印DEBUG级别的信息）
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')

    # 文件打印设置（默认使用logger的级别设置，当达到2MB时对日志文件分文件存储）
    file_handler = handlers.RotatingFileHandler(filename=settings.LOG_PATH, maxBytes=2097152)
    file_handler.setFormatter(formatter)

    # 添加设置
    my_log.addHandler(chlr)
    my_log.addHandler(file_handler)

    return my_log

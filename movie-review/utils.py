# -*- coding: utf-8 -*-
"""
工具栏模块
"""

import settings


def read_vocab_list():
    """
    读取词汇表并返回
    :return:
    """
    with open(settings.VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab_list = f.read().strip().split('\n')
    return vocab_list


def read_word_to_id_dict():
    """
    生成词到字典ID的映射
    :return:
    """
    vocab_list = read_vocab_list()
    word2id = dict(zip(vocab_list, range(len(vocab_list))))
    return word2id


def get_id_by_word(word, word2id):
    """
    获取词在词典中的ID
    :param word: 词
    :param word2id: 词典，若不存在返回 word2id['<unkown>']
    :return:
    """
    if word in word2id:
        return word2id[word]
    else:
        return word2id['<unkown>']

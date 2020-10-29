# -*- coding: utf-8 -*-
"""
数据预处理
"""
import collections
import numpy as np

import utils
import settings


def decode_file(infile, outfile, decodes='Windows-1252'):
    """
    将文件的编码转换为Unicode
    :param decodes: 解码类型
    :param infile: 输入文件路径
    :param outfile: 输出文件路径
    :return:
    """
    with open(infile, 'rb') as f:
        txt = f.read().decode(encoding=decodes)
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(txt)


def get_encode_info(file):
    """
    判断文本编码
    :param file: 输入文本路径
    :return:
    """
    import chardet
    with open(file, 'rb') as f:
        data = f.read()
        encoding = chardet.detect(data)
    return encoding['encoding']


def create_vocab():
    """
    创建词汇表，并写入文件中
    :return:
    """
    # 存放出现的所有单词
    word_list = []
    # 从文件中读取数据，拆分单词
    with open(settings.NEG_TXT, 'r', encoding='utf-8') as f:
        f_lines = f.readlines()
        for line in f_lines:
            words = line.strip().split()
            word_list.extend(words)
    with open(settings.POS_TXT, 'r', encoding='utf-8') as f:
        f_lines = f.readlines()
        for line in f_lines:
            words = line.strip().split()
            word_list.extend(words)

    # 统计单词出现的次数
    counter = collections.Counter(word_list)

    # 选取高频词
    sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    word_list = [word[0] for word in sorted_words]
    word_list = ['<unkown>'] + word_list[:settings.VOCAB_SIZE - 1]

    # 将词汇表写入文件中
    with open(settings.VOCAB_PATH, 'w', encoding='utf-8') as f:
        for word in word_list:
            f.write(word + '\n')


def create_vec(txt_path, vec_path):
    """
    以句子为单位，将词汇表转换为词向量形式并存储
    :param txt_path: 解码后的源文件
    :param vec_path: 输出词向量路径
    :return:
    """
    # 获取单词到编号的映射
    word2id = utils.read_word_to_id_dict()

    # 将语句转化成向量
    vec = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tmp_vec = [str(utils.get_id_by_word(word, word2id)) for word in line.strip().split()]
            vec.append(tmp_vec)

    # 写入文件中
    with open(vec_path, 'w', encoding='utf-8') as f:
        for tmp_vec in vec:
            f.write(' '.join(tmp_vec) + '\n')


def shuffle_data(x, y, path):
    """
    补全数据及填充数据，最后写入np文件中
    :param x: 数据
    :param y: 标签
    :param path: 保存路径
    :return:
    """
    # 计算每一数据集的最大长度
    maxlen = max(map(len, x))

    # 填充数据
    data = np.zeros([len(x), maxlen], dtype=np.int32)
    for row in range(len(x)):
        data[row, :len(x[row])] = x[row]
    label = np.array(y)

    # 打乱数据
    state = np.random.get_state()
    np.random.shuffle(data)

    np.random.set_state(state)
    np.random.shuffle(label)

    # 保存数据
    np.save(path + '_data', data)
    np.save(path + '_labels', label)


def cut_train_dev_test():
    """
    划分训练集、开发集和测试集
    :return:
    """
    # 三个子列表分别存放训练、开发、测试
    data = [[], [], []]
    labels = [[], [], []]

    # 累加概率 rate [0.8, 0.1, 0.1]  cumsum_rate [0.8, 0.9, 1.0]
    rate = np.array([settings.TRAIN_RATE, settings.DEV_RATE, settings.TEST_RATE])
    cumsum_rate = np.cumsum(rate)

    # 划分数据集
    with open(settings.POS_VEC, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tmp_data = [int(word) for word in line.strip().split()]
            tmp_label = [1, ]
            # 以事先设定好的概率区间进行随机放置数据
            index = int(np.searchsorted(cumsum_rate, np.random.rand(1) * 1.0))
            data[index].append(tmp_data)
            labels[index].append(tmp_label)

    with open(settings.NEG_VEC, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tmp_data = [int(word) for word in line.strip().split()]
            tmp_label = [0, ]
            index = int(np.searchsorted(cumsum_rate, np.random.rand(1) * 1.0))
            data[index].append(tmp_data)
            labels[index].append(tmp_label)

    # 计算一下实际上分割出来的比例
    datas = list(map(len, data))
    print('最终分割比例', np.array([datas], dtype=np.float32) / sum(datas))

    # 打乱数据，写入到文件中
    shuffle_data(data[0], labels[0], settings.TRAIN_DATA)
    shuffle_data(data[1], labels[1], settings.DEV_DATA)
    shuffle_data(data[2], labels[2], settings.TEST_DATA)


if __name__ == '__main__':
    # 获取编码信息
    neg_encoding = get_encode_info(settings.ORIGIN_NEG)
    pos_encoding = get_encode_info(settings.ORIGIN_POS)

    # 解码数据集
    decode_file(settings.ORIGIN_NEG, settings.NEG_TXT, neg_encoding)
    decode_file(settings.ORIGIN_POS, settings.POS_TXT, pos_encoding)

    # 创建词汇表
    create_vocab()

    # 生成词向量
    create_vec(settings.NEG_TXT, settings.NEG_VEC)
    create_vec(settings.POS_TXT, settings.POS_VEC)

    # 划分数据集
    cut_train_dev_test()

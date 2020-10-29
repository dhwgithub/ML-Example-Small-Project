# -*- coding: utf-8 -*-
"""
定义模型
"""
import tensorflow as tf
import functools

import settings


def doublewrap(function):
    """
    自定义define_scope前置处理
    :param function:
    :return:
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    自定义scope修饰器
    :param function:
    :param scope:
    :param args:
    :param kwargs:
    :return:
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class Model(object):
    def __init__(self, data, label, emb_keep, rnn_keep):
        self.data = data
        self.label = label
        self.emb_keep = emb_keep
        self.rnn_keep = rnn_keep
        # 其他必要属性值
        self.predict
        self.loss
        self.global_step
        self.ema
        self.optimize
        self.acc

    @define_scope
    def predict(self):
        """
        定义前向传播过程
        :return:
        """
        # 词嵌入矩阵权重
        embedding = tf.get_variable('embedding', [settings.VOCAB_SIZE, settings.HIDDEN_SIZE])

        # 使用dropout的LSTM
        lstm_cell = [
            tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(settings.HIDDEN_SIZE), self.rnn_keep)
            for _ in range(settings.NUM_LAYERS)
        ]

        # 构建循环神经网络
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)

        # 生成词嵌入矩阵，并进行dropout
        inputs = tf.nn.embedding_lookup(embedding, self.data)
        dropout_input = tf.nn.dropout(inputs, self.emb_keep)

        # 计算rnn的输出
        outputs, last_state = tf.nn.dynamic_rnn(cell, dropout_input, dtype=tf.float32)

        # 做二分类问题，这里只需要最后一个节点的输出
        last_output = outputs[:, -1, :]

        # 求最后节点输出的线性加权和
        weights = tf.Variable(tf.truncated_normal([settings.HIDDEN_SIZE, 1]), dtype=tf.float32, name='weights')
        bias = tf.Variable(0, dtype=tf.float32, name='bias')

        logits = tf.matmul(last_output, weights) + bias

        return logits

    @define_scope
    def ema(self):
        """
        定义移动平均
        :return:
        """
        ema = tf.train.ExponentialMovingAverage(settings.EMA_RATE, self.global_step)
        return ema

    @define_scope
    def loss(self):
        """
        定义损失函数，这里使用交叉熵
        :return:
        """
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.predict)
        loss = tf.reduce_mean(loss)
        return loss

    @define_scope
    def global_step(self):
        """
        注意指定trainable=False
        :return:
        """
        global_step = tf.Variable(0, trainable=False)
        return global_step

    @define_scope
    def optimize(self):
        """
        定义反向传播过程
        :return:
        """
        # 学习率衰减
        learn_rate = tf.train.exponential_decay(settings.LEARN_RATE,
                                                self.global_step,
                                                settings.LR_DECAY_STEP,
                                                settings.LR_DECAY)
        # 反向传播优化器
        optimizer = tf.train.AdamOptimizer(learn_rate).minimize(self.loss, global_step=self.global_step)

        # 移动平均操作
        ave_op = self.ema.apply(tf.trainable_variables())

        # 组合构成训练op
        with tf.control_dependencies([optimizer, ave_op]):
            train_op = tf.no_op('train')

        return train_op

    @define_scope
    def acc(self):
        """
        定义模型acc计算过程
        :return:
        """
        # 对前向传播的结果求sigmoid
        output = tf.nn.sigmoid(self.predict)

        # 真负类
        ok0 = tf.logical_and(tf.less_equal(output, 0.5), tf.equal(self.label, 0))

        # 真正类
        ok1 = tf.logical_and(tf.greater(output, 0.5), tf.equal(self.label, 1))

        # 一个数组，所有预测正确的都为True,否则False
        ok = tf.logical_or(ok0, ok1)

        # 先转化成浮点型，再通过求平均来计算acc
        acc = tf.reduce_mean(tf.cast(ok, dtype=tf.float32))

        return acc


if __name__ == '__main__':
    x = tf.placeholder(tf.int32, [8, 20])
    y = tf.placeholder(tf.float32, [8, 1])
    model = Model(x, y, 0.8, 0.8)

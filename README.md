# ML-Example-Small-Project
机器学习实例小项目 - 使用 Python 3 和 Tensorflow 开发

## 一、Movie-review
电影评论情感分析，使用 Python 3 和 tensorflow 1 进行开发学习，参考实现：https://github.com/AaronJny/emotional_classification_with_rnn （对该项目的学习，详细教程请查看原项目）

本项目对如上项目进行了 python 3 版本的学习和实践，修正了部分在 python 3 中无法运行的错误（如文件读取等），注：tf.nn.dynamic_rnn 方法已被弃用

### 使用方法
- 解压 datasets 中的压缩包，将两个文件放置于 datasets 目录中
- 运行 process_data.py 进行数据预处理
- 运行 train.py 开始训练
- eval.py 用于验证和测试
- 所有配置信息都存放于 settings.py

## 二、Poetry-generator
古诗生成器，使用了 Python 3 和 tensorflow 2 进行开发学习，参考实现：https://github.com/AaronJny/DeepLearningExamples/tree/master/tf2-rnn-poetry-generator （对该项目的学习，同时对原项目进行了部分修改，加入了日志打印和保存模块，对某些方法进行了新增注释）

原项目基础上新增日志功能模块，当文件大小满 2M 时自动分文件存储日志信息

### 使用方法
- 运行 train.py 开始训练
- 运行 eval.py 对训练后的模型进行效果检验
- 运行过程中的时间及关键步骤信息等存于 logs/poetry.logs 中
- 所有配置信息都存放于 settings.py

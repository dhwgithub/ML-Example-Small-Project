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

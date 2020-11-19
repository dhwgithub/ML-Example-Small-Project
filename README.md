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

## 三、Object_detection_by_faster_rcnn 
目标检测小实例，使用 Python 3 和 tensorflow 1 进行开发学习，参考实现：https://blog.csdn.net/qq_36758914/article/details/105886811 （详细讲解请查看原博客，感谢！！）

对源码进行了一定的修改，同时增加了易于自己理解的注释。最后时间有限，仅训练了 3/10 的量，但是还是能看出来有效果的

### 使用方法
- 运行 train.py 开始训练
- 运行 test.py 对图像进行目标检测，之后将标出的图像保存到指定位置
- 注意训练数据访问和原博客不一致，但是数据集是一样的
- 对数据进行了人为拆分，达到了图像名不再受约束的目的

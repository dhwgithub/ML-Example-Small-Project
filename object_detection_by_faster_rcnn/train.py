"""
参考博客：https://blog.csdn.net/qq_36758914/article/details/105886811
"""
import tensorflow as tf

import utils
from rpn import RPNplus
import setting

# 创建数据生成器，包含images, target_scores, target_bboxes, target_masks
trainSet = utils.DataGenerator(setting.train_dataset_path, setting.batch_size)

# 创建模型和优化器
model = RPNplus()
optimizer = tf.keras.optimizers.Adam(lr=setting.learn_rate)

# 记录日志
writer = tf.summary.create_file_writer("./log")

# 全局步数
global_steps = tf.Variable(0, trainable=False, dtype=tf.int64)

for epoch in range(setting.EPOCHS):
    for step in range(setting.STEPS):
        global_steps.assign_add(1)
        image_data, target_scores, target_bboxes, target_masks = next(trainSet)

        with tf.GradientTape() as tape:
            pred_scores, pred_bboxes = model(image_data)
            score_loss, boxes_loss = utils.compute_loss(target_scores,
                                                        target_bboxes,
                                                        target_masks,
                                                        pred_scores,
                                                        pred_bboxes)
            total_loss = score_loss + setting.lambda_scale * boxes_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print("=> epoch %d  step %d  total_loss: %.6f  score_loss: %.6f  boxes_loss: %.6f"
                  % (epoch+1, step+1, total_loss.numpy(), score_loss.numpy(), boxes_loss.numpy()))

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("total_loss", total_loss, step=global_steps)
            tf.summary.scalar("score_loss", score_loss, step=global_steps)
            tf.summary.scalar("boxes_loss", boxes_loss, step=global_steps)
        writer.flush()

    model.save_weights("RPN.h5")
'''
可视化训练损失
tensorboard --logdir=./log
'''

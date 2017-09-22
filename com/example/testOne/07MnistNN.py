# coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 获取mnist数据集
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 操作符号变量来描述这些可交互的操作单元,占位符
x = tf.placeholder(tf.float32, [None, 784])  # 2维的浮点数张量(?, 784)

# 初始化w、b，可修改的张量,待学习的值
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 实现简单模型,神经网络
pred = tf.nn.softmax(tf.matmul(x, w) + b)

# 添加一个新的占位符用于输入正确值
y = tf.placeholder(tf.float32, [None, 10])
# 计算交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))

# 训练模型
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# # 初始化所有变量
# tf.global_variables_initializer().run()
# # 训练模型1000次
# for i in range(1000):
#     # 随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     train_step.run({x: batch_xs, y: batch_ys})
#
#
# # 评估我们的模型
# correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

# 初始化所有变量
init = tf.global_variables_initializer()
# 启动一个会话
with tf.Session() as sess:
    sess.run(init)
    # 训练模型1000次
    for i in range(1000):
        # 随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
    # 评估我们的模型
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))  # 0.92

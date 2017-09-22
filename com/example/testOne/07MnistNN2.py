# coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 获取mnist数据集
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 使用InteractiveSession类
sess = tf.InteractiveSession()

# 定义两个占位符
x = tf.placeholder(tf.float32, [None, 784])  # 2维的浮点数张量(?, 784)
y = tf.placeholder(tf.float32, [None, 10])

# 变量：初始化w、b，可修改的张量,待学习的值
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 初始化所有变量
sess.run(tf.initialize_all_variables())

# 实现简单模型,神经网络
pred = tf.nn.softmax(tf.matmul(x, w) + b)

# 计算交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))

# 训练模型
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 训练模型1000次
for i in range(1000):
    # 每一步迭代，我们都会加载50个训练样本，然后执行一次train_step，
    # 并通过feed_dict将x 和 y_张量占位符用训练训练数据替代。
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y: batch[1]})
# 评估我们的模型
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))  # 0.87

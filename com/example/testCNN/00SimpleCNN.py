# coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 获取数据集
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
sess = tf.InteractiveSession()


# 定义权重函数
def weight_varible(shape):
    # 给权重制造一些随机噪声打破完全对称，截断的正态分布噪声
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


# 定义偏置项函数
def bias_varible(shape):
    # 由于使用ReLU，也给偏置增加一些小的正值（0.1）
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


# 定义卷积函数
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 定义池化函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 设计CNN之的结构前，定义两个占位符
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 定义所有的网络参数
weights = {
    'wc1': weight_varible([5, 5, 1, 32]),
    'wc2': weight_varible([5, 5, 32, 64]),
    'wfc1': weight_varible([7*7*64, 1024]),
    'out': weight_varible([1024, 10]),
}
biases = {
    'bc1': bias_varible([32]),
    'bc2': bias_varible([64]),
    'bfc1': bias_varible([1024]),
    'out': bias_varible([10]),
}

# 第一层卷积层
conv1 = tf.nn.relu(conv2d(x_image, weights['wc1']) + biases['bc1'])
pool1 = max_pool_2x2(conv1)

# 第二层卷积层
conv2 = tf.nn.relu(conv2d(pool1, weights['wc2']) + biases['bc2'])
pool2 = max_pool_2x2(conv2)

# 全连接层
pool_fc1 = tf.reshape(pool2, [-1, 7*7*64])
fc1 = tf.nn.relu(tf.matmul(pool_fc1, weights['wfc1']) + biases['bfc1'])

# 减轻过拟合，使用dropout层，是通过placeholder传入keep_prob比率来控制
keep_prob = tf.placeholder(tf.float32)
fc1_drop = tf.nn.dropout(fc1, keep_prob)

# 输出层
# Dropout层的输出连接一个softmax层，得到最后概率输出
y_conv = tf.nn.softmax(tf.matmul(fc1, weights['out']) + biases['out'])

# 损失函数， 优化器
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 评价准确率
correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# bool值转化为float32值，求平均
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 训练模型
tf.global_variables_initializer().run()

for i in range(20000):
    batch = mnist.train.next_batch(100)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    # 92%



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


# 定义权重函数
def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


# 定义偏置项函数
def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


# 定义卷积函数
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 定义池化函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第一层卷积层
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积层
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# 为了减少过拟合，我们在输出层之前加入dropout。
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

pred = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# 训练和评估模型
# 计算交叉熵
cross_entropy = -tf.reduce_sum(y * tf.log(pred))

# 训练模型
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 评估我们的模型
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess.run(tf.global_variables_initializer())

# 训练模型1000次
for i in range(20000):
    # 每一步迭代，我们都会加载50个训练样本，然后执行一次train_step，
    # 并通过feed_dict将x 和 y_张量占位符用训练训练数据替代。
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
    print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))  # 0.87


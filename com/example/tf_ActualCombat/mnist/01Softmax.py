# coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 1、加载数据
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 2、构建模型
# 设置训练超参数
lr = 0.001
training_iters = 100000
batch_size = 128
'''
为了使用RNN来分类图片，把每张图片的行看成是一个像素序列（ sequence ）
MNIST 图片的大小是 28 × 28 像素,所以把每一个图像样本看成一行行的序列,共有(28 个元素的序列）×（28 行），
然后每一步输入的序列长度是 28 ， 输入的步数是 28 步
'''

# 神经网络的参数
n_inputs = 28  # 输入层的n
n_steps = 28  # 28长度
n_hidden = 128  # 隐藏层的神经元个数
n_classes = 10  # 输出的数量，即分类的类别，0～9个数字，共有10个

# 定义输入数据及权重
# 输入数据占位符
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])
# 定义权重
weights = {
	'in': tf.Variable(tf.random_normal([n_inputs, n_hidden]))
}










# 定义回归模型
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, w) + b  # 预测值

# 定义损失函数和优化器
y_ = tf.placeholder(tf.float32, [None, 10])
# 用 tf.nn.softmax_cross_entropy_with_logits 来计算预测值y与真实值y_的差值，并取均值
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
# 采用 SGD 作为优化器
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 3、训练模型
'''
使用 InteractiveSession （）来创建交互式上下文的 TensorFlow 会话
＃与常规会话不同的是，交互式会话会成为默认会话
'''
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 随机训练:训练1000次，每次抓取100个数据点
for _ in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	
# 评估训练好的模型
# 计算预测值和其实值
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 布尔型转化为浮点数，并取平均值 ， 得到准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 计算模型在测试集上的准确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))  # 0.9165

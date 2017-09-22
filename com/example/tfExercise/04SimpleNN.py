# coding=utf-8
import tensorflow as tf
import numpy as np

# 要学习的方程为 y=x^2-0.5,构造满足这个方程的一堆 x 和 y，同时加入一些不满足方程的噪声点
# 1、生成及加载输入数据
# 构造满足一元二次方程的函数
'''
为了使点更密一些，我们构建了300个点，分布在－1 到 1 区间，直接采用 np 生成等差数列的方法，
并将结果为 300 个点的一维数组，转换为 300 × 1 的二维数组
'''
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# 加入一些噪声点 ，使它与x_data的维度－致，并且拟合为均值为0、方差为0.05的正态分布
noise = np.random.normal(0, 0.05, x_data.shape)
# y = x^2 - 0.5 ＋噪声
y_data = np.square(x_data) - 0.5 + noise
# 定义x和y的占位符来作为将要输入神经网络的变量
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


# 2、构建网络模型
# 输入数据、输入数据的维度、输出数据的维度和激活函数
def add_layer(inputs, in_size, out_size, activate_function=None):
	# 构建权重：in_size x out_size大小的矩阵
	weights = tf.Variable(tf.random_normal([in_size, out_size]))
	# 构建偏置：1 x out_size大小的矩阵
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	# 矩阵相乘
	Wx_plus_b = tf.matmul(inputs, weights) + biases
	if activate_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activate_function(Wx_plus_b)
	# 得到输出数据
	return outputs

# 构建隐藏层，假设隐藏层有 10 个神经元
h1 = add_layer(xs, 1, 20, activate_function=tf.nn.relu)
# 构建输出层，假设输出层和输入层一样，有 1 个神经元
prediction = add_layer(h1, 20, 1, activate_function=None)

# 计算预测值和真实值间的误差
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=1))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 3、训练模型
# TensorFlow 训练 1000 次，每 50 次输出训练的损失值：
init = tf.global_variables_initializer()  # 初始化所有变盘
with tf.Session() as sess:
	sess.run(init)
	for i in range(1000):
		sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
		if i % 50 == 0:
			print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
'''
训练出权重值来使模型拟合 y =x^2- 0.5 的系数 1 和-0.5 ，通过损失值越来越小，
可以看出训练参数越来越逼近目标结果。
'''

# coding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
'''
每个图像样本的行 看成 一个像素序列
共有( 28 个元素 的序列）×（ 28 行），
然后每一步输入的序列长度是 28 ， 输入的步数是 28 步。
'''
# 1、加载数据
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 2、构建模型

# 设置训练的超参数
lr = 0.001
training_iters = 100000
batch_size = 128

# 神经网络的参数
n_inputs = 28  # 输入层n
n_steps = 28  # 28 长度
n_hidden = 128  # 隐藏层的神经元个数
n_classes = 10  # 输出的数量，即分类的类别，0~9个数字，共有10个

# 定义输入数据集权重
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 定义权重
weights = {
	# (28, 128)
	'in': tf.Variable(tf.random_normal([n_inputs, n_hidden])),
	# (128, 10)
	'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
	# (28, )
	'in': tf.Variable(tf.constant(0.1, shape=[n_hidden, ])),
	# (10, )
	'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


# 定义RNN模型
def RNN(x, weights, biases):
	# 原始的 x 是 3 维数据, 需要把它变成 2 维数据才能使用 weights 的矩阵乘法
	# reshape:x——>(128 batch * 28 steps , 28 inputs)
	x = tf.reshape(x, [-1, n_inputs])
	
	# 进入隐藏层
	# x_in = w*x + b(128 batch * 28 steps , 28 inputs)
	x_in = tf.matmul(x, weights['in']) + biases['in']
	# x_in ——> (128 batch * 28 steps , 28 inputs)
	x_in = tf.reshape(x_in, [-1, n_steps, n_hidden])
	
	# 采用基本的 LSTM 循环网络单元 ： basic LSTM Cell
	lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
	
	# 初始化为零值， lstm 单元由两个部分组成 ： （ c_state , h_state )
	init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
	
	'''
		tf.nn.dynamic_rnn(cell, inputs), 要确定 inputs 的格式.
		tf.nn.dynamic_rnn 中的 time_major 参数会针对不同 inputs 格式有不同的值.
		 inputs 为 (batches, steps, inputs) ==> time_major=False;
		 inputs 为 (steps, batches, inputs) ==> time_major=True;

	'''
	# dynamic_rnn 接收张量（batch, steps, inputs）或者（steps ,batch ,inputs ）作为x_in
	outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=init_state, time_major=False)
	
	results = tf.matmul(final_state[1], weights['out']) + biases['out']
	return results

# 定义损失函数和优化器 ， 优化器采用 AdamOptimizer
predict = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

# 定义模型预测结果及准确率计算方法
correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 3、 训练、数据及评估模型
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 随机训练:训练100000次, 每次抓取128个数据点
step = 0
while step * batch_size < training_iters:
	batch_xs, batch_ys = mnist.train.next_batch(batch_size)
	batch_xs = batch_xs.reshape([batch_size, n_steps,n_inputs])
	sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys})
	if step % 20 == 0:
		print("count ", step, ": ", sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))
	step += 1

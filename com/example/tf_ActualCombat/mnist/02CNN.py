# coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

# 1、加载数据:FLAGS.data_dir 是 MNIST 所在的路径
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
x_train, y_train, x_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# (55000, 784), (55000, 10), (10000, 784), (10000, 10)
# 处理输入的数据：形状变为 ［－1, 28, 28, 1]， -1 表示不考虑输入图片的数量，
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 定义两个占位符
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

# 2、初始化权重与定义网络结构


# 定义初始化权重函数
def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))
# 初始化权重,设置卷积核的大小为3×3
wc1 = init_weights([3, 3, 1, 32])  # patch 大小为 3 × 3 ，输入维度为 1 ，输出维度为 32
wc2 = init_weights([3, 3, 32, 64])  # patch 大小为 3 × 3 ，输入维度为 32 ，输出维度为 64
wc3 = init_weights([3, 3, 64, 128])  # patch 大小为 3 × 3 ，输入维度为 64 ，输出维度为 128
wf1 = init_weights([128*4*4, 625])  # 全连接层，输入维度为 128 × 4 x 4 ， 是上一层的输出数据又三维的转变成一维， 输出维度为 625
wo1 = init_weights([625, 10])  # 输出层，输入维度为 625 ， 输出维度为 10 ，代表 10 类（ labels)


# 定义网络结构
# 神经网络模型的构建函数 ， 传入以下参数
# x: 输入数据
# w ： 每一层的权重
# p_keep_conv, p_keep_hidden: dropout 要保留的神经元比例
def model(x, wc1, wc2, wc3, wf1, wo1, p_keep_conv , p_keep_hidden):
	# 第一组
	l1a = tf.nn.relu(tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='SAME'))  # (?,28, 28, 32)
	l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # (?,14, 14, 32)
	l1 = tf.nn.dropout(l1, p_keep_conv)
	
	# 第二组
	l2a = tf.nn.relu(tf.nn.conv2d(l1, wc2, strides=[1, 1, 1, 1], padding='SAME'))  # (?,14, 14, 64)
	l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # (?,7, 7, 64)
	l2 = tf.nn.dropout(l2, p_keep_conv)
	
	# 第三组
	l3a = tf.nn.relu(tf.nn.conv2d(l2, wc3, strides=[1, 1, 1, 1], padding='SAME'))  # (?,7, 7, 128)
	l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # (?,4, 4, 128)
	l3 = tf.reshape(l3, [-1, wf1.get_shape().as_list()[0]])  # (?, 2048)
	l3 = tf.nn.dropout(l3, p_keep_conv)
	
	# 第四层
	l4 = tf.nn.relu(tf.matmul(l3, wf1))
	l4 = tf.nn.dropout(l4, p_keep_hidden)
	
	# 输出层
	pyx = tf.matmul(l4, wo1)
	return pyx  # 预测值

# 定义 dropout 的占位符
p_keep_conv = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)
py_x = model(x, wc1, wc2, wc3, wf1, wo1, p_keep_conv, p_keep_hidden)  # 得到预测值

# 定义损失函数和优化器
# 用 tf.nn.softmax_cross_entropy_with_logits 来计算预测值y与真实值y_的差值，并取均值
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=y))
# 采用RMSProp作为优化器:学习率为0.001， 衰减值为0.9，使损失最小
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)
predict_op = tf.argmax(py_x, 1)

# 3、训练模型和评估模型
# 定义训练时的批次大小和评估时的批次大小
batch_size = 128
test_size = 256
# 在一个会话中启动图 ， 开始训练和评估
with tf.Session() as sess:
	# 初始化所有变量
	tf.global_variables_initializer().run()
	# 随机训练:训练100次
	for i in range(100):
		# [(0, 128), (128, 256), (256, 384), (384, 512), (512, 640), ......]
		training_batch = zip(range(0, len(x_train), batch_size), range(batch_size, len(x_train)+1, batch_size))
		for start, end in training_batch:
			sess.run(train_step, feed_dict={x: x_train[start: end], y: y_train[start: end],
																			p_keep_conv: 0.8, p_keep_hidden: 0.5})
			
		test_indices = np.arange(len(x_test))  # 得到一个test batch[0, ... , 9999]
		np.random.shuffle(test_indices)  # 打乱列表
		test_indices = test_indices[0: test_size]
		print(i, np.mean(np.argmax(y_test[test_indices], axis=1) ==
										 sess.run(predict_op, feed_dict={x: x_test[test_indices],
																										 p_keep_conv: 1.0,
																										 p_keep_hidden: 1.0})))

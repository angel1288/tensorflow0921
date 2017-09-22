# coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
'''
无监督学习：无标记
'''
# 1、加载数据
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 2、构建模型
# 设置训练的超参数
learning_rate = 0.01  # 学习率
training_epochs = 20  # 训练的轮数
batch_size = 256  # 每次训练的数据多少
display_step = 1  # 每隔多少轮显示一次训练结果

# 参数
examples_to_show = 10  # 从测试集中选择 10 张图片去验证自动编码器的结果
# 网络参数
n_hidden_1 = 256  # 第一个隐藏层神经元个数，也是特征值个数
n_hidden_2 = 128  # 第二个隐藏层神经元个数，也是特征值个数
n_input = 784  # 输入数据的特征值个数： 28 × 28=784

# 定义输入图像
x = tf.placeholder(tf.float32, [None, n_input])

# 初始化每一层的权重和偏置
weights = {
	'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
	'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biaes = {
	'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# 定义自动编码模型的网络结构，包括压缩和解压两个过程 ：
# 定义压缩函数
def encoder(x):
	# Encoder Hidden layer with sigmoid activation #1
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biaes['encoder_b1']))
	# Encoder Hidden layer with sigmoid activation #2
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biaes['encoder_b2']))
	return layer_2


# 定义解压函数
def decoder(x):
	# Decoder Hidden layer with sigmoid activation #1
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biaes['decoder_b1']))
	# Decoder Hidden layer with sigmoid activation #2
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biaes['decoder_b2']))
	return layer_2

# 构建模型
encoder_op = encoder(x)
decoder_op = decoder(encoder_op)

# 3、构建损失函数和优化器
# 得出预测值
y_pred = decoder_op
# 得出其实值，即输入值
x_true = x
# 定义损失函数和优化器
cost = tf.reduce_mean(tf.pow(x_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# 3、训练数据及评估模型
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
total_batch = int(mnist.train.num_examples / batch_size)
# 开始训练
for epoch in range(training_epochs):
	
	for i in range(total_batch):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		# Run optimization op (backprop) and cost op (t o get loss value)
		_, c = sess.run([optimizer, cost], feed_dict={x: batch_xs})
	# 每一轮，打印出一次损失值
	if epoch % display_step == 0:
		print("Epoch:", '%04d' % (epoch+1), "cost= ", "{:.9f}".format(c))
print("Optimization Finished !")

# 对测试集应用训练好的自动编码网络
encode_decode = sess.run(y_pred, feed_dict={x: mnist.test.images[:examples_to_show]})
# 比较测试集原始图片和自动编码网络的重建结果
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
	a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))  # 测试集
	a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))  # 重建结果
f.show()
plt.draw()
plt.waitforbuttonpress()

# coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 下载mnist数据集
mnist = input_data.read_data_sets('/tmp/', one_hot=True)

# 图像28x28，RNN将数据分块输入到网络
chunk_size = 28
chunk_n = 28

rnn_size = 256

n_output_layer = 10  # 输出层
# 定义两个占位符
X = tf.placeholder(tf.float32, [None, chunk_n, chunk_size])
Y = tf.placeholder(tf.float32)


# 定义待训练的神经网络
def recurrent_neural_network(data):
	layer = {'w_': tf.Variable(tf.random_normal([rnn_size, n_output_layer])),
					 'b_': tf.Variable(tf.random_normal([n_output_layer]))}
	
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
	
	# 将最外两层转置
	# 将输入数据由[ batch_size,nsteps,n_inputs] 变为 [ nsteps，batch_size,n_inputs]
	data = tf.transpose(data, [1, 0, 2])
	data = tf.reshape(data, [-1, chunk_size])
	data = tf.split(0, chunk_n, data)
	outputs, status = tf.nn.rnn(lstm_cell, data, dtype=tf.float32)
	
	output = tf.add(tf.matmul(outputs[-1], layer['w_']), layer['b_'])
	
	return output

# 每次使用100条数据进行训练
batch_size = 100


# 使用数据训练神经网络
def train_neural_network(X, Y):
	predict = recurrent_neural_network(X)
	cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, Y))
	optimizer = tf.train.AdamOptimizer().minimize(cost_func)
	
	epochs = 13
	with tf.Session() as session:
		session.run(tf.initialize_all_variables())
		epoch_loss = 0
		for epoch in range(epochs):
			for i in range(int(mnist.train.num_examples / batch_size)):
				x, y = mnist.train.next_batch(batch_size)
				x = x.reshape([batch_size, chunk_n, chunk_size])
				_, c = session.run([optimizer, cost_func], feed_dict={X: x, Y: y})
				epoch_loss += c
			print(epoch, ' : ', epoch_loss)
			
			correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
			print('准确率: ', accuracy.eval({X: mnist.test.images.reshape(-1, chunk_n, chunk_size), Y: mnist.test.labels}))
		
train_neural_network(X, Y)
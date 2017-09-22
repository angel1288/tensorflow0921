# coding=utf-8
import matplotlib.pyplot as plt
import pandas as pd
import requests
import io
import numpy as np
import tensorflow as tf

url = 'http://blog.topspeedsnail.com/wp-content/uploads/2016/12/铁路客运量.csv'
ass_data = requests.get(url).content

# 读取CSV（逗号分割）文件到DataFrame中
df = pd.read_csv(io.StringIO(ass_data.decode('utf-8')))  # 读入铁路客运量数据
data = np.array(df['铁路客运量_当期值(万人)'])  # 获取'铁路客运量_当期值(万人)'序列

# 归一化处理：np.std 计算矩阵的标准差
normalized_data = (data - np.mean(data)) / np.std(data)

# plt.figure()
# plt.plot(data)
# plt.show()

seq_size = 3  # 时间步
train_x, train_y = [], []  # 训练集 [-1, seq_size, input_dim]
for i in range(len(normalized_data) - seq_size - 1):
	train_x.append(np.expand_dims(normalized_data[i : i + seq_size], axis=1).tolist())
	train_y.append(normalized_data[i + 1: i + seq_size + 1].tolist())
	
input_dim = 1  # 维度
X = tf.placeholder(tf.float32, [None, seq_size, input_dim])  # 每批次输入网络的tensor[?, 3, 1]
Y = tf.placeholder(tf.float32, [None, seq_size])  # 每批次tensor对应的标签[?, 3]


# 回归
def ass_rnn(hidden_layer_size=6):
	w = tf.Variable(tf.random_normal([hidden_layer_size, 1]), name='w')
	b = tf.Variable(tf.random_normal([1]), name='b')
	
	cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size)
	# cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size)
	# outputs输出：[batch_size, max_time, cell.output_size]
	outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
	
	# 通过拼接一个给定的tensor构建一个新的tensor
	w_repeated = tf.tile(tf.expand_dims(w, 0), [tf.shape(X)[0], 1, 1])
	out = tf.matmul(outputs, w_repeated) + 1
	out = tf.squeeze(out)
	return out


# 训练RNN
def train_rnn():
	out = ass_rnn()
	# 损失函数
	loss = tf.reduce_mean(tf.square(out - Y))
	train_op = tf.train.AdamOptimizer(learning_rate=0.003).minimize(loss)
	
	saver = tf.train.Saver(tf.global_variables())
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# 重复训练10000次
		for step in range(10000):
			_, loss_ = sess.run([train_op, loss], feed_dict={X: train_x, Y: train_y})
			if step % 10 == 0:
				# 用测试数据评估loss
				print(step, loss_)
		print("保存模型: ", saver.save(sess, 'E:/ImageRecognition/TensorflowExample/com/example/tfExercise/ass.model'))
	
		
# 预测
def prediction():
	out = ass_rnn()
	
	saver = tf.train.Saver(tf.global_variables())
	with tf.Session() as sess:
		saver.restore(sess, './ass.model')
		
		# 取训练集最后一行为测试样本。shape=[1, 3, 1]
		prev_seq = train_x[-1]
		predict = []
		# 得到之后2个预测结果
		for i in range(12):
			next_seq = sess.run(out, feed_dict={X: [prev_seq]})
			predict.append(next_seq[-1])
			# 每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
			prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
			
			plt.figure()
			plt.plot(list(range(len(normalized_data))), normalized_data, color='b')
			plt.plot(list(range(len(normalized_data), len(normalized_data) + len(predict))), predict, color='r')
			plt.show()

# train_rnn()
prediction()

# ValueError: Variable rnn/basic_lstm_cell/kernel already exists, disallowed.
# 当train和predict放在一起的时候，会调用两次class language_model

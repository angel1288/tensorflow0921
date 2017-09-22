# encoding=utf-8
import tensorflow as tf
import numpy as np

print("1------------------------")
input_data = tf.Variable(np.random.rand(10, 6, 6, 3), dtype=np.float32)
fliter_data = tf.Variable(np.random.rand(2, 2, 3, 10), dtype=np.float32)
y = tf.nn.conv2d(input_data, fliter_data, strides=[1, 1, 1, 1], padding='SAME')
print(y)  # 输出：Tensor("Conv2D:0", shape=(10, 6, 6, 10), dtype=float32)
output = tf.nn.avg_pool(value=y, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
print(output)  # 输出：Tensor("AvgPool:0", shape=(10, 6, 6, 10), dtype=float32)

print("2------------------------")
input_data = tf.Variable(np.random.rand(10, 6, 6, 3), dtype=np.float32)
fliter_data = tf.Variable(np.random.rand(2, 2, 3, 10), dtype=np.float32)
y = tf.nn.conv2d(input_data, fliter_data, strides=[1, 1, 1, 1], padding='SAME')
output = tf.nn.max_pool(value=y, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
print(output)  # 输出：Tensor("MaxPool:0", shape=(10, 6, 6, 10), dtype=float32)
with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    print(sess.run(output))  # 输出其值
    print(sess.run(tf.shape(output)))  # 输出：[10  6  6 10]

print("3------------------------")
input_data = tf.Variable(np.random.rand(10, 6, 6, 3), dtype=np.float32)
fliter_data = tf.Variable(np.random.rand(2, 2, 3, 10), dtype=np.float32)
y = tf.nn.conv2d(input_data, fliter_data, strides=[1, 1, 1, 1], padding='SAME')
output, argmax = tf.nn.max_pool_with_argmax(input=y, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
print(y)  # 输出：Tensor("Conv2D_2:0", shape=(10, 6, 6, 10), dtype=float32)
print(output)  # 输出：Tensor("MaxPoolWithArgmax:0", shape=(10, 6, 6, 10), dtype=float32)
print(argmax)  # 输出：Tensor("MaxPoolWithArgmax:1", shape=(10, 6, 6, 10), dtype=int64)
# GPU下
with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    print(sess.run(output))
    print(sess.run(tf.shape(output)))
    print(sess.run(argmax))
    print(sess.run(tf.shape(argmax)))


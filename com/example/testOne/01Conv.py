# encoding=utf-8
import tensorflow as tf
import numpy as np

print("1--------------------------------")
# 产生一个10*9维数组，数组中每个元素为9*3维矩阵，其中每个值在[0, 1)范围内（0-1正态分布）
input_data = tf.Variable(np.random.rand(10, 9, 9, 3), dtype=np.float32)
fliter_data = tf.Variable(np.random.rand(2, 2, 3, 2), dtype=np.float32)
y = tf.nn.conv2d(input_data, fliter_data, strides=[1, 1, 1, 1], padding='SAME')
print(tf.shape(y))  # 输出：Tensor("Shape:0", shape=(4,), dtype=int32)
# 输出通道的总数是channel_multiplier=2
print(y)  # 输出：Tensor("Conv2D:0", shape=(10, 9, 9, 2), dtype=float32)

print("2--------------------------------")
input_data = tf.Variable(np.random.rand(10, 9, 9, 3), dtype=np.float32)
fliter_data = tf.Variable(np.random.rand(2, 2, 3, 5), dtype=np.float32)
y = tf.nn.depthwise_conv2d(input_data, fliter_data, strides=[1, 1, 1, 1], padding='SAME')
print(tf.shape(y))  # 输出：Tensor("Shape_1:0", shape=(4,), dtype=int32)
# 输出通道的总数是(in_channel=3） * （channel_multiplier=5）
print(y)  # 输出：Tensor("depthwise:0", shape=(10, 9, 9, 15), dtype=float32)

print("3--------------------------------")
input_data = tf.Variable(np.random.rand(10, 9, 9, 3), dtype=np.float32)
depthwise_filter = tf.Variable(np.random.rand(2, 2, 3, 5), dtype=np.float32)
pointwise_filter = tf.Variable(np.random.rand(1, 1, 15, 20), dtype=np.float32)
# out_channels >= channel_multiplier * in channels
y = tf.nn.separable_conv2d(input_data, depthwise_filter, pointwise_filter, strides=[1, 1, 1, 1], padding='SAME')
print(tf.shape(y))  # 输出：Tensor("Shape_2:0", shape=(4,), dtype=int32)
print(y)  # 输出：Tensor("separable_conv2d:0", shape=(10, 9, 9, 20), dtype=float32)

print("4--------------------------------")
input_data = tf.Variable( np.random . rand(1, 5, 5, 1), dtype=np.float32)
filters = tf.Variable( np.random.rand(3, 3, 1, 1), dtype=np.float32)
y = tf.nn.atrous_conv2d(input_data, filters, 2, padding='SAME')
print(tf.shape(y))  # 输出：Tensor("Shape_3:0", shape=(4,), dtype=int32)
print(y)  # 输出：Tensor("convolution/BatchToSpaceND:0", shape=(1, 5, 5, 1), dtype=float32)

print("5--------------------------------")
x = tf.random_normal(shape=[1, 3, 3, 1])
kernel = tf.random_normal(shape=[2, 2, 3, 1])
y = tf.nn.conv2d_transpose(x, kernel, output_shape=[1, 5, 5, 3], strides=[1, 2, 2, 1], padding='SAME')
print(tf.shape(y))  # 输出：Tensor("Shape_4:0", shape=(4,), dtype=int32)
print(y)  # 输出：Tensor("conv2d_transpose:0", shape=(1, 5, 5, 3), dtype=float32)

# encoding=utf-8
import tensorflow as tf
import numpy as np

print("1--------------------------------")
# 矩阵乘法
# 创建图
a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])
c = a * b
# 创建会话
sess = tf.Session()
# 计算c
print(sess.run(c))
sess.close()

print("2--------------------------------")
# 创建一个常量运算操作，产生一个 1 × 2 矩阵
mat1 = tf.constant([[3., 3.]])
# 创建另外一个常量运算操作，产生一个 2 × 1 矩阵
mat2 = tf.constant([[2.], [2.]])
# 创建一个矩阵乘法运算，把 mat1 和 mat2 作为输入
result = tf.matmul(mat1, mat2)
# 创建会话
sess = tf.Session()
# 计算result
print(sess.run([result]))
sess.close()

print("3--------------------------------")
a = tf.constant([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
sess = tf.Session()
print(sess.run(tf.sigmoid(a)))
# 输出：[[ 0.7310586   0.88079703]
#       [ 0.7310586   0.88079703]
#       [ 0.7310586   0.88079703]]

print("4--------------------------------")
a = tf.constant([-1.0, 2.0])
with tf.Session() as sess:
    b = tf.nn.relu(a)
    print(sess.run(b))
# 输出：[ 0.  2.]

print("5--------------------------------")
a = tf.constant([[-1.0, 2.0, 3.0, 4.0]])
with tf.Session() as sess:
    b = tf.nn.dropout(a, 0.5, noise_shape=[1, 4])
    print(sess.run(b))
    b = tf.nn.dropout(a, 0.5, noise_shape=[1, 1])
    print(sess.run(b))
# 输出： [[-0.  4.  0.  0.]]
#       [[-0.  0.  0.  0.]]

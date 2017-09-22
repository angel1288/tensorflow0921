# encoding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os


def init_weights(shape):
    # 生成正态分布随机数，均值mean=0.0,标准差stddev=0.01
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
    # 第一个全连接层
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_keep_hidden)
    # 第二个全连接层
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h, p_keep_hidden)

    # 输出预测值
    return tf.matmul(h2, w_o)

# 1、加载及定义模型
# 加载数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# #tf.placeholder可以理解为形参，用于定义过程，在执行的时候再赋具体的值
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

# 初始化权重参数
w_h = init_weights([784, 625])
w_h2 = init_weights([625, 625])
w_o = init_weights([625, 10])

# 生成网络模型，得到预测值
p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

# 定义损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# 2、训练模型及存储模型
# 定义一个存储路径：当前路径下的 ckpt_dir 目录
ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# 定义一个计数器：为训练轮数计数
# #计数器变量，设置它的 trainable=False ，不需要被训练
global_step = tf.Variable(0, name='global_step', trainable=False)

# 在声明完所有变最后，调用tf.train.Saver
saver = tf.train.Saver()
# 位于tf.train.Saver 之后的变量将不会被存储
non_storable_varible = tf.Variable(777)

# 训练模型并存储
with tf.Session() as sess:
    # 变量初始化
    # sess.run(tf.initialize_all_variables())
    tf.initialize_all_variables().run()
    # 得到 global_step 的初始值
    start = global_step.eval()
    print("Start from: ", start)

    for i in range(start, 100):
        # 以 128 作为 batch_size
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})

        # 更新计数器
        global_step.assign(i).eval()
        # 存储模型
        saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)

        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX,
                                                         p_keep_input: 1.0,
                                                         p_keep_hidden: 1.0})))

# with tf.Session() as sess :
#     tf.initialize_all_variables().run()
#     ckpt = tf.train.get_checkpoint_state(ckpt_dir)
#     if ckpt and ckpt.model_checkpoint_path :
#         print(ckpt.model_checkpoint_path)
#         # 加载所有的参数
#         saver.restore(sess, ckpt.model_checkpoint_path)
#         # 从这里开始就可以直接使用模型进行预测，或者接着继续训练了

# with tf.Session() as sess:
#     with gfile.FastGFile("/tmp/tfmodel/train.pbtxt", 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFrornString(f.read())
#         _sess.graph.as_default()
#         tf.import_graph_def(graph_def, name='tfgraph')


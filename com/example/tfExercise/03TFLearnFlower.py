# coding=utf-8
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tflearn.datasets.oxflower17 as oxflower17

# 使用第三方库实现AlexNet
# 牛津大学的鲜花数据集:17个类别的鲜花数据，
# 每个类别 80 张图片，并且图片有大量的姿态和光的变化
X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

# 构建AlexNet网络模型
network = input_data(shape=[None, 227, 227, 3])  # 输入：?x227x227x3
network = conv_2d(network, 96, 11, strides=4, activation='relu')  # fliter：96x96x3x11, 4, 输出：?x57x57x96
network = max_pool_2d(network, 3, strides=2)											# kernel：3x3, 2, 输出：?x29x29x96
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')							# fliter：5x5x96x256, 1, 输出：?x29x29x256
network = max_pool_2d(network, 3, strides=2)											# kernel：3x3, 2, 输出：?x15x15x256
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')							# fliter：3x3x256x384, 1, 输出：?x15x15x384
network = conv_2d(network, 384, 3, activation='relu')							# fliter：3x3x384x384, 1, 输出：?x15x15x384
network = conv_2d(network, 256, 3, activation='relu')							# fliter：3x3x384x256, 1, 输出：?x15x15x256
network = max_pool_2d(network, 3, strides=2)											# kernel：3x3, 2, 输出：?x8x8x256
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')				# 输出：?x4096
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')				# 输出：?x4096
network = dropout(network, 0.5)
network = fully_connected(network, 17, activation='softmax')			# 输出：?x17
# ＃回归操作，同时规定网络所使用的学习率、损失函数和优化器
network = regression(network, optimizer='momentum', loss='categorical_crossentropy', learning_rate=0.001)

# 训练模型
model = tflearn.DNN(network, checkpoint_path='model_alexnet', max_checkpoints=1, tensorboard_verbose=2)
model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True, show_metric=True, batch_size=64, snapshot_step=200,
					snapshot_epoch=False, run_id='alexnet_oxflower17')

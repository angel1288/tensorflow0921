# coding=utf-8
import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Activation, Dropout, Flatten
from keras.datasets import mnist
from keras import backend as K

# 超参数
batch_size = 128
nb_classes = 10  # 分类数
nb_epoch = 12  # 训练轮数


# 输入图像的维度
img_rows, img_cols = 28, 28
# 卷积滤镜的个数
nb_filters = 32
# 最大池化，池化核大小
pool_size = (2, 2)
# 卷积核大小
kernel_size = (3, 3)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
	# 使用Theano的顺序：(conv_diml , channels , conv_dim2 , conv_dim3)
	x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	input_shape = (1, img_rows, img_cols)
else:
	# 使用Tensorflow的顺序：(conv_diml, conv_dim2 , conv_dim3, channels )
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

# 类型转换
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 将类向量转换成二进制类矩阵
y_train = keras.utils.to_categorical(y_train, nb_classes)
y_test = keras.utils.to_categorical(y_test, nb_classes)

# 构建模型：用 2 个卷积层、 1 个池化层和 2 个全连接层来构建
'''
Keras 的核心数据结构是模型。模型是用来组织网络层的方式。型有两种:
1、Sequential模型是一系列网络层按顺序构成的栈，是单输入和单输出的，层与层之间只有相邻关系，是最简单的一种模型。
2、Model模型：建立更复杂的模型。
'''
model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size=kernel_size[0], strides=kernel_size[1], padding='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(x_test, y_test))

# 评估模型: 计算出的损失值和准确率
score = model.evaluate(x_train, y_train, verbose=0)
print('Test score :', score[0])
print('Test accuracy:', score[1])

# 模型的加载及保存






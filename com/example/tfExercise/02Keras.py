# coding=utf-8
from keras.models import Sequential
from keras.layers import Dense, activations

# Sequ ential 模型是一系列网络层按顺序构成的拢，是单
# 输入和单输出的， 层与层之间 只有相邻关系 ， 是最简单的一种模型 。
model = Sequential()
model.add(Dense(output_dim=64, input_dim=100))
model.add(activations("relu"))
model.add(Dense(output_dim=10))
model.add(activations("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd ', metrics=['accuracy'])
model.fit(X_train)

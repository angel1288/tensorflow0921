# encoding=utf-8
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist
import os
import argparse
import sys

FLAGS = None


# 1、生成 TFRecords 文件
# #定义主函数，给训练、验证、测试数据集做转换
def main(unused_argv):
    # 获取数据
    data_sets = mnist.read_data_sets(FLAGS.directory, dtype=tf.uint8,
                                     reshape=False, validation_size=FLAGS.validation_size)
    # 将数据转换为 tf.train.Example 类型，并写入 TFRecords 文件
    convert_to(data_sets.train, 'train')
    convert_to(data_sets.validation, 'validation')
    convert_to(data_sets.test, 'test')


# 转换数据集为tfrecords
def convert_to(data_set, name):
    # 55000个训练数据，5000个验证数据，10000个测试数据
    images = data_set.images
    labels = data_set.labels
    num_examples = data_set.num_examples

    if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d .' % (images.shape[0], num_examples))
    rows = images.shape[1]  # 28
    cols = images.shape[2]  # 28
    depth = images.shape[3]  # 1是黑白图像（单通道）

    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(num_examples):
        image_raw = images[index].tostring()

        # 写入协议缓冲区中，height、width、depth、label编码成int64类型，image_raw编码成二进制
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        # 序列化为字符串
        writer.write(example.SerializeToString())
    writer.close()


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    # argparse是python用于解析命令行参数和选项的标准模块
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='/tmp/data',
                        help='Directory to download data files and write the converted result')
    parser.add_argument('--validation_size', type=int, default=5000,
                        help="""\
                        Number of examples to separate from the training data for the validationset.\
                        """)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
# 输出：
# Extracting /tmp/data\train-images-idx3-ubyte.gz
# Extracting /tmp/data\train-labels-idx1-ubyte.gz
# Extracting /tmp/data\t10k-images-idx3-ubyte.gz
# Extracting /tmp/data\t10k-labels-idx1-ubyte.gz
# Writing /tmp/data\train.tfrecords
# Writing /tmp/data\validation.tfrecords
# Writing /tmp/data\test.tfrecords


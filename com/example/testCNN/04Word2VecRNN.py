# coding=utf-8
import collections
import math, random, zipfile, urllib, os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 定义下载文本数据的函数
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        # 将远程数据下载到本地
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    # 返回相关文件的系统状态信息
    statinfo = os.stat(filename)
    # 文件的大小，以字节为单位
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        # 使用raise抛出异常:一旦执行了raise语句，raise后面的语句将不能执行
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browers?')
    return filename

filename = maybe_download('text8.zip', 31344016)


# 定义解压下载的压缩文件的函数
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename)
print('Data size', len(words))  # 17005207

# 创建vocabulary词汇表
vocabulary_size = 50000


def build_dataset(words):
    count = [['UNK', -1]]  # len(count)=1,表示只有一组数据；UNK=unknown
    # Counter：字典的子类，用于统计哈希对象; most_common 返回一个TopN列表
    # 收集全部的单词的词频,按照词频（出现的次数）从高到低排序（添加到count后面）
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    # 创建一个字典
    dictionary = dict()
    # 将全部单词转为编号（以频数排序的编号），top50000之外的单词，
    # 认为UnKown,编号为0,并统计这类词汇的数量
    for word, _ in count:
        # 键是单词，值是索引；按照键的字典顺序排序
        dictionary[word] = len(dictionary)
    # 创建一个列表
    data = list()
    unk_count = 0
    for word in words:  # 遍历词汇列表
        # 对于其中每一个词汇，先判断是否出现在dictionary中，
        if word in dictionary:
            # 如果出现，则转为其编号
            index = dictionary[word]
        else:
            # 如果不是，则转为编号0
            index = 0
            unk_count += 1  # 统计unknow的词汇个数
        data.append(index)
    count[0][1] = unk_count
    # zip()接受任意>=0个序列作为参数，返回一个tuple列表。
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)

# 删除原始单词列表，节省内存
del words
# 打印vocabulary中最高频出现的词汇及其数量
print('Most common words (+UNK)', count[:5])
# Most common words (+UNK) [['UNK', 418391], ('the', 1061396), ('of', 593677), ('and', 416629), ('one', 411764)]
# dtata中前10个词汇及其对应编号
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
# Sample data [5234, 3081, 12, 6, 195, 2, 3134, 46, 59, 156] ['anarchism', 'originated',
#  'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against']

# 生成word2vec训练样本
data_index = 0


# 定义生成训练用的batch数据函数：batch_size，skip_window单词最远可以联系的距离；
# num_skips对每个单词生成多少个样本（<=skip_window的两倍，且batch_size必须是其整数倍），
# 确保每个batch包含了一个词汇对应的all样本
def generate_batch(batch_size, num_skips, skip_window):
    # 设为全局变量，确保它可在函数generate_batch中被修改
    global data_index
    # assert确保num_skips和batch_size满足以下条件
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    # 将batch和labels初始化为数组
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)  # 1行batch_size列，行向量
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)  # batch_size行1列，列向量
    # 定义span为对某个单词创建相关样本时会使用到的单词数量，包括目标单词本身和它前后的单词
    span = 2 * skip_window + 1
    # deque为了高效实现插入和删除操作的双向列表，适合用于队列和栈
    # 创建一个最大容量为span的deque
    buffer = collections.deque(maxlen=span)

    # 以data_index开始，把span个单词顺序读入buffer作为初始值
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # 第一层循环每次循环内对一个目标单词生成样本，buffer中是目标单词和all相关单词
    for i in range(batch_size // num_skips):
        # 目标单词
        target = skip_window
        # 定义生成样本时需要avoid的单词列表，初始包含目标单词（目的：预测语境单词）
        targets_to_avoid = [skip_window]
        # 第二层循环，每次循环对一个语境生成样本
        for j in range(num_skips):
            # 产生随机数，直至随机数不在targets_to_avoid中，
            # 代表can use的语境单词，then产生一个样本，feature即目标词汇buffer[skip_window],
            # label即 buffer[target]
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]  # feature
            labels[i * num_skips + j] = buffer[target]  # label
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

# # 简单测试generate_batch的功能
# batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
# for i in range(8):
#     print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
#           reverse_dictionary[labels[i, 0]])
# # 3081 originated -> 5234 anarchism
# # 3081 originated -> 12 as
# # 12 as -> 3081 originated
# # 12 as -> 6 a
# # 6 a -> 195 term
# # 6 a -> 12 as
# # 195 term -> 6 a
# # 195 term -> 2 of

# 定义训练参数
batch_size = 128
embedding_size = 128  # 将单词转为稠密向量的维度，一般在50~1000范围内，此处选用128作为词向量的维度
skip_window = 1
num_skips = 2
# 生成验证数据valid_examples，随机抽取频数最高的单词，看向量上是否跟他们最近的单词是否相关性比较高
valid_size = 16  # 抽取的验证单词数
valid_window = 100  # 验证单词从频数最高的100个单词中抽取，随机抽取
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64  # 训练时用来做负样本的噪声单词的数量


# ###########定义Skip-Gram Word2Vec模型的网络结构##############
# 创建一个tf.Graph图
graph = tf.Graph()
with graph.as_default():  # 设置为默认图
    # 设置两个占位符
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    # 将前面的valid_examples转化为TF中的constant
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    # 限定所有计算在CPU上执行，有些操作GPU无法实现
    with tf.device('/cpu:0'):
        # 随机生成所有单词的词向量embeddings[5000，128]，值的大小[-1.0, 1.0]：正态分布随机数
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # 在train_inputs中查找train_inputs对应的表示，对应的向量embed
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        # 使用NCE Loss作为训练的优化目标
        nce_weights = tf.Variable(  # 产生截断正态分布随机数
            tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    # NCE Loss，计算学习出的词向量在训练数据上的loss，并使用tf.reduce_mean汇总
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=train_labels,
                                         inputs=embed,
                                         num_sampled=num_sampled,
                                         num_classes=vocabulary_size))

    # 定义优化器SGD，且学习率为1.0
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # #######计算between minibatch examples and all embeddings的余弦相似性#############
    # 计算嵌入向量embeddings的L2范数norm
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    # 将embeddings除以其L2范数得到标准后的normalized_embeddings
    normalized_embeddings = embeddings / norm
    # 查询验证单词的嵌入向量
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    # 计算验证单词的嵌入向量与词汇表中所有单词的相似性
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # 初始化所有模型参数
    init = tf.global_variables_initializer()

# ############训练#############
# 定义最大迭代次数，创建并设置默认session
num_steps = 100001

with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized")

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # 执行一次优化器运算和损失运算，并将训练的loss累积到average_loss上
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        # 每2000次循环，计算一下平均loss并显示出来
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # 每10000次循环，计算一次验证单词与全部单词的相似度，并将与每个验证单词最相似的8个单词展示出来
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()


# #############定义可视化Word2Vec效果的函数########################
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    # 新建绘画窗口,独立显示绘画的图片
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        # 散点图
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        # 文本注释
    # 保存图像
    plt.savefig(filename)

# 使用sklearn.manifold.TSNE实现降维128 to 2
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
# 用plot_with_labels展示词频最高的100个单词的可视化结果
plot_only = 100
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)








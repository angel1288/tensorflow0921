# encoding=utf-8
import tensorflow as tf

# 创建一个先进先出的队列，初始队列插入0.1、0.2/0/3三个数
q = tf.FIFOQueue(3, "float")
init = q.enqueue_many(([0.1, 0.2, 0.3], ))
# 定义出队、+1、入队操作
x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])

# 开启一个会话，执行2次q_inc 操作，随后查看队列的内容
with tf.Session() as sess:
    sess.run(init)
    quelen = sess.run(q.size())
    for i in range(2):
        # 执行2次操作，队列中的值变为 0.3 , 1.1 , 1.2
        sess.run(q_inc)

    quelen = sess.run(q.size())
    for i in range(quelen):
        # 输出队列的值
        print(sess.run(q.dequeue()))
# 输出：0.3
#      1.1
#      1.2
print("-----------------------------")

# 创建一个随机队列，队列最大长度为10，出队后最小长度为2
q = tf.RandomShuffleQueue(capacity=10, min_after_dequeue=2, dtypes="float")
# 开启一个会话，执行10次入队操作，8次出队操作
with tf.Session() as sess:
    # 10次入队
    for i in range(0, 10):
        sess.run(q.enqueue(i))

    # 10次入队
    for i in range(0, 8):
        print(sess.run(q.dequeue()))
# 输出随机，每次不同
# 1.0   9.0
# 3.0   2.0
# 2.0   7.0
# 9.0   4.0
# 0.0   5.0
# 5.0   8.0
# 8.0   6.0
# 6.0   3.0
# run_options = tf.RunOptions(timeout_in_ms=10000)  # 等待 10 秒
# try:
#     sess.run(q.dequeue(), options=run_options)
# except tf.errors.DeadlineExceededError:
#     print('out of range')
print("2-------------------------------")

# 1、创建一个含有队列的图
q = tf.FIFOQueue(1000, "float")
# 计数器
counter = tf.Variable(0.0)
# 操作：给计数器加1
increment_op = tf.assign_add(counter, tf.constant(1.0))
# 计数器值加入队列
enqueue_op = q.enqueue(counter)

# 2、创建一个队列管理器 QueueRunner，用这两个操作向队列 q 中添加元素，仅用一个线程
qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enqueue_op] * 1)
# # 3、启动一个会话，从队列管理器 qr 中创建线程
# # 主线程
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     # 启动入队线程
#     enqueue_threads = qr.create_threads(sess, start=True)
#     # 主线程
#     for i in range(10):
#         print(sess.run(q.dequeue()))
print("3-------------------------------")

# 3、使用协调器（ coordinator）来管理线程
# 主线程
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Coordinator ：协调器，协调线程间的关系可以视为一种信号量，用来做同步
    coord = tf.train.Coordinator()
    # 启动入队线程，协调器是线程的参数
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

    # 主线程
    # for i in range(0, 10):
    #     print(sess.run(q.dequeue()))
    # # 通知其他线程关闭
    # coord.request_stop()
    # # join 操作等待其他线程结束，其他所有线程关闭之后，这一函数才能返回
    # coord.join(enqueue_threads)

    coord.request_stop()
    # 主线程
    for i in range(0, 10):
        try:
            print(sess.run(q.dequeue()))  # 输出2.0（较多）或1.0
        except tf.errors.OutOfRangeError:
            break
    coord.join(enqueue_threads)
print("4-------------------------------")

# 预加载数据
x1 = tf.constant([2, 3, 4])
x2 = tf.constant([4, 0, 0])
y = tf.add(x1, x2)
print(y)  # 输出:Tensor("Add:0", shape=(3,), dtype=int32)
with tf.Session() as sess:
    print(sess.run(y))  # 输出:[6 3 4]
print("5-------------------------------")

# 使用sess.run()中的feed_dict参数，将Python产生的数据填充给后端
# 设计图
a1 = tf.placeholder(tf.int16)
a2 = tf.placeholder(tf.int16)
b = tf.add(a1, a2)
# 用Python产生数据
l1 = [2, 3, 4]
l2 = [4, 0, 1]
# 打开一个会话，将数据填充给后端
with tf.Session() as sess:
    print(sess.run(b, feed_dict={a1: l1, a2: l2}))  # 输出:[6 3 5]
print("6-------------------------------")

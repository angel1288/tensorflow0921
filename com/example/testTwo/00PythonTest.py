# coding=utf-8
import math


# 斐波拉契数列推算
def fib(n):
    m, a, b, = 0, 0, 1
    while m < n:
        yield b
        a, b = b, a + b
        m += 1

fib(4).__next__()
for s in fib(4):
    print(s)


# 删除1-100之间是素数的值
def not_prime(x):
    if x <= 1:
        return
    for i in range(2, int(math.sqrt(x)) + 1):
        if x % i == 0:
            return True
    return False
x = filter(not_prime, range(1, 101))
y = []
for i in x:
    y.append(i)
print(y)


# 保留1-100之间是素数的值
def is_prime(x):
    if x <= 1:
        return
    for i in range(2, int(math.sqrt(x)) + 1):
        if x % i == 0:
            return False
    return True
x = filter(is_prime, range(1, 101))
y = []
for i in x:
    y.append(i)
print(y)


def is_odd(n):
    return n % 2 == 1
x = filter(is_odd, [1, 2, 4, 5, 6, 9, 10, 15])
y = []
for i in x:
    y.append(i)
print(y)
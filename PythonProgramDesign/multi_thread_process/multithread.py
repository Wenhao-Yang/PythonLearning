#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: multithread.py
@Time: 2019/9/25 上午11:25
@Overview:
"""
import threading
import time

num_thread = 0

# 这个函数名可随便定义
def run(n):
    global num_thread
    num_thread += 1
    print("current task start：{}. And {} threads are running!".format(n, num_thread))
    time.sleep(1)
    print("current task stop：", n)

class MyThread(threading.Thread):

    def __init__(self, n):
        super(MyThread, self).__init__()  # 重构run函数必须要写
        self.n = n

    def run(self):
        global num_thread
        num_thread += 1
        for i in range(0, 10):
            time.sleep(1)
            print('thread {} is working in {}%.'.format(self.n, 100* (i+1)/10))
#
# if __name__ == "__main__":
#     t1 = threading.Thread(target=run, args=("thread 1",))
#     t2 = threading.Thread(target=run, args=("thread 2",))
#     t3 = threading.Thread(target=run, args=("thread 3",))
#     t1.start()
#     t2.start()
#     t3.start()

    # t1.join()
    # t2.join()
    # t3.join()

if __name__ == "__main__":
    # num_thread = 0
    t1 = MyThread("thread 1")
    t2 = MyThread("thread 2")
    threadpool = []
    threadpool.append(t1)
    threadpool.append(t2)
    for t in threadpool:
        t.start()

    t1.join()
    t2.join()

    print('end.')


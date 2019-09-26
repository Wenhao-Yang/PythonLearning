#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: multiprocess.py
@Time: 2019/9/25 下午4:21
@Overview:
"""
import time
from multiprocessing import Process, Queue

def get_from_queue(queue, pid):

    while not queue.empty():
        cont = queue.get()
        time.sleep(1)
        print('Thread {} get {}, and queue has {} left.'.format(pid, cont, queue.qsize()))

if __name__ == '__main__':
    queue = Queue()
    for i in range(0, 15):
        queue.put(i)

    pro = Process(target=get_from_queue, args=(queue, 1))
    pro2 = Process(target=get_from_queue, args=(queue, 2))
    pro.start()
    pro2.start()

    #print(queue.get())
    pro.join()
    pro2.join()
    #print(queue.get())

# class MyProcess(Process):
#     def __init__(self, name):
#         super(MyProcess, self).__init__()
#         self.name = name
#
#     def run(self):
#         print('process name :' + str(self.name))
#         time.sleep(1)
#
# if __name__ == '__main__':
#
#     for i in range(3):
#         p = MyProcess(str(i))
#         p.start()
#     for i in range(3):
#         p.join()
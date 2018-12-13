#!/usr/bin/python3

import sys

def fibonacci(n):
    a,b,counter=0,1,0
    while True:
        if (counter>n):
            return
        yield a
        a,b=b,a+b
        counter += 1

f=fibonacci(10)
while True:
    try:
        print(next(f), end=" ")
    except StopIteration:
        sys.exit()


def printinfo1(arg1, *vartuple):
    """
    加了星号 * 的参数会以元组(tuple)的形式导入，存放所有未命名的变量参数。

    """
    print(arg1)
    for var in vartuple:
        print(var)
    return

printinfo1(1,2,3)
"""
1
2
3
"""

def printinfo2(arg1,**var_args_dict):
    """

    加了两个星号 ** 的参数会以字典的形式导入以传入不定长的参数；

    """
    print(arg1)
    print(var_args_dict)
    return
printinfo2(1,a=2,b=3)
"""
1
{'a': 2, 'b': 3}
"""
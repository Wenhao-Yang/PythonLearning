cdef extern from "demo.cpp":
    pass

# Decalre the class with cdef
cdef extern from "demo.h" namespace "demo":
    cdef cppclass MyDemo:
        MyDemo() except +
        MyDemo(int) except +
        int a
        int mul(int )
        int add(int )
        void sayHello(char*)
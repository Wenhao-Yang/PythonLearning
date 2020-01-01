# distutils: language = c++

from cdemo cimport MyDemo

# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.
cdef class PyMyDemo:
    cdef MyDemo c_mydemo  # Hold a C++ instance which we're wrapping

    def __cinit__(self,a):
        self.c_mydemo = MyDemo(a)
    def mul(self, m):
        return self.c_mydemo.mul(m)

    def add(self,b):
        return self.c_mydemo.add(b)

    def sayHello(self,name ):
        self.c_mydemo.sayHello(name)

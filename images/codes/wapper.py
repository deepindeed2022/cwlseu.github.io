import time

# class Timeit(object):
#     def __init__(self, func):
#         self._wrapped = func

#     def __call__(self, *args, **kws):
#         start_time = time.time()
#         result = self._wrapped(*args, **kws)
#         print("elapsed time is %s " % (time.time() - start_time))
#         return result

class Timeit(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = self.func(*args, **kwargs)
        print("elapsed time is %s " % (time.time() - start_time))
        return result

    def __get__(self, instance, owner):
        print("invoking Timeit.__get__")
        start_time = time.time()
        time.sleep(1)
        result = lambda *args, **kwargs: self.func(instance, *args, **kwargs)    
        print("elapsed time is %s " % (time.time() - start_time))
        return result

@Timeit
def func():
    time.sleep(1)
    return"invoking function func"

class A(object):
    @Timeit
    def func(self):
        return "invoking function A.func"

if __name__ == '__main__':
    a = A()
    print a.func()# Boom!
    print 
    print func()

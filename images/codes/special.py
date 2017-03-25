# !/usr/bin/python 
# -*- coding: utf-8 -*- 
class Singleton(object):
	_instance = None
	def __new__(self, *args, **kwargs):
		if self._instance:
			return self._instance
		self._instance = cv = object.__new__(self, *args, **kwargs)
		return cv

sin1 = Singleton()
sin2 = Singleton()
print(sin1 is sin2)
print Singleton() is sin2

# class SingleMeta(type):
#     def __init__(cls, name, bases, dict):
#         cls._instance = None
#         __new__o = cls.__new__

#         def __new__(cls, *args, **kwargs):
#             if cls._instance:
#                 return cls._instance
#             cls._instance = cv = __new__o(cls, *args, **kwargs)
#             return cv
#         cls.__new__ = __new__
# class A(object):
#     __metaclass__ = SingleMeta
# a1 = A()  # what`s the fuck

class TraceAttribute(type):
    def __init__(cls, name, bases, dict):
        __getattribute__o = cls.__getattribute__

        def __getattribute__(self, *args, **kwargs):
            print('__getattribute__:', args, kwargs)
            return __getattribute__o(self, *args, **kwargs)
        cls.__getattribute__ = __getattribute__

class A(object):  # Python 3 是 class A(object,metaclass=TraceAttribute):
    __metaclass__ = TraceAttribute
    a = 1
    b = 2
a = A()
a.a
a.b

class SingleMeta(type):
	def __init__(self, name, bases, dict):
		self._instance = None
		__new__o = self.__new__

		@staticmethod
		def __new__(self, *args, **kwargs):
			if self._instance:
				return self._instance
			self._instance = cv = __new__o(self, *args, **kwargs)
			return cv
		self.__new__ = __new__

class A(object):
	__metaclass__ = SingleMeta

a1 = A() # what`s the fuck
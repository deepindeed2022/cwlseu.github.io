# class A(object):
# 	def __call__(self):
# 		print("invoking __call__ from A!")

# if __name__ == "__main__":
# 	a = A()
# 	a()# output: invoking __call__ from A
# 	a.__call__ = lambda: "invoking __call__ from lambda"
# 	print a.__call__()
# 	a()

class C(object):
	'''There is comment
	'''
	pass
	def __len__(self):
		return 5

c = C()
c.__len__ = lambda: 5
print c.__dict__
print type(c).__dict__
print len(c)
# class Grade(object):
# 	def __init__(self):
# 		import weakref
# 		self._grade_pool = weakref.WeakKeyDictionary()
# 	def __get__(self,instance,owner):
# 		return self._grade_pool.get(instance,None)

# 	def __set__(self,instance,value):
#  		if 0 <= value <= 100:
# 			_grade_pool = self.__dict__.setdefault('_grade_pool',{})
# 			_grade_pool[instance] = value
# 		else:
# 			raise ValueError("fuck")

class Grade(object):
	def __get__(self, instance, owner):
		return instance.__dict__[self.key]

	def __set__(self, instance, value):
		if 0 <= value <= 100:
			instance.__dict__[self.key] = value
		else:
			raise ValueError("fuck")
	def __set_name__(self, owner, name):
		self.key = name

class Exam(object):
	math = Grade()

	def __init__(self, math):
		self.math = math

if __name__ == '__main__':
	niche = Exam(math = 90)
	print(niche.math)
	# output : 90
	snake = Exam(math = 75)
	print(snake.math)
	# output : 75
	try:
		snake.math = 120
	except ValueError as e:
		print e
	print niche.math
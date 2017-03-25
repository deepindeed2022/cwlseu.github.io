# class Grade(object):
# 	def __init__(self):
# 		self._score = 0

# 	def __get__(self, instance, owner):
# 		return self._score

# 	def __set__(self, instance, value):
# 		if 0 <= value <= 100:
# 			self._score = value
# 		else:
# 			raise ValueError('grade must be between 0 and 100')

class Grade(object):
	def __init__(self):
		self._grade_pool = {}

	def __get__(self, instance, owner):
		return self._grade_pool.get(instance, None)

	def __set__(self, instance, value):
		if 0 <= value <= 100:
			_grade_pool = self.__dict__.setdefault('_grade_pool',{})
			_grade_pool[instance] = value
		else:
			raise ValueError("Ooh, Value Error")

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
	# output: ValueError:grade must be between 0 and 100!
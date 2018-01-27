# -*- encoding:utf-8 -*-
from __future__ import division
import math
import operator


# 组合数函数
def C(n, k):
	if k > n or k < 0: return 0
	elif k == 0: return 1
	elif n - k < k:
		return C(n, n-k)
	else:
		return  reduce(operator.mul, range(n - k + 1, n + 1)) \
				/reduce(operator.mul, range(1, k +1))  

# 泊松分布
def posson(lambda_, k):
	return (lambda_**k/math.factorial(k))*math.exp(-lambda_)

# 伯努利二项分布
def bernonlli(p, n=1, k=1):
	return C(n, k)*(p**k)*((1-p)**(n-k))

# gamma函数是阶乘在实数上的延拓，gamma(x+1) = x*gamma(x)
def gamma(x):
	if x <= 0:
		raise ValueError("Gamma Parameter Error!")
	if x == 0.5:
		return math.pi**0.5
	else:
		return (x-1)*gamma(x-1)

def gaussion(x, mean=0, std_variance=1):
	return 1/(math.sqrt(2*math.pi*std_variance)) \
			* math.exp(-((x - mean)**2)/(2*std_variance))
def test_gaussion():
	total = 0
	for i in range(-1, 1):
		total += gaussion(i)
	print total
def test_bernonlli():
	assert bernonlli(0.5) == 0.5
	assert bernonlli(0.5, 2, 1) == 0.5

	assert bernonlli(0.5, 3, 0) == 1/8
	assert bernonlli(0.5, 3, 1) == 3/8
	assert bernonlli(0.5, 3, 2) == 3/8
	assert bernonlli(0.5, 3, 3) == 1/8
	assert bernonlli(0.5, 3, 4) == 0

def test_C():
	assert C(10, 2) == 45
	assert C(100, 10) == C(100, 90)
	assert C(100000, 1) == 100000
	assert C(100000, 0) == 1

if __name__ == '__main__':
	test_C()
	test_bernonlli()
	assert math.gamma(0.5) == math.pi**0.5
	#test_gaussion()
import numpy as np
import pdb
''' 
In this problem we implement gradient descent on an error function
E(u, v) = (ue^v - 2ve^-u)^2 with a learning rate of 0.1
'''

def error(u, v):
	u = float(u)
	v = float(v)
	return pow(u * np.exp(v) - 2 * v * np.exp(-u), 2)

def error_deriv_u(u, v):
	u = float(u)
	v = float(v)
	return 2 * (np.exp(v) + 2 * v * np.exp(-u)) * (u * np.exp(v) - 2 * v * np.exp(-u))

def error_deriv_v(u, v):
	u = float(u)
	v = float(v)
	return 2 * (u * np.exp(v) - 2 * np.exp(-u)) * (u * np.exp(v) - 2 * v * np.exp(-u))

def gradient_descent(threshold):
	u, v = (1, 1)
	eta = 0.1
	_iter = 0
	while (error(u, v) > threshold):
		gradient = (eta * error_deriv_u(u, v), eta * error_deriv_v(u, v))
		u, v = np.subtract((u, v), gradient)
		_iter += 1
	return (u, v, _iter)

def coordinate_descent(iterations):
	u, v = (1, 1)
	eta = 0.1
	for _ in xrange(iterations):
		u = u - eta * error_deriv_u(u, v)
		v = v - eta * error_deriv_v(u, v)
	return (u, v)

def grad_descent_problem():
	u, v, _iter = gradient_descent(pow(10, -14))
	print "num iterations w/ gradient descent:", _iter
	print "final (u, v) =", (u, v)

	u, v = coordinate_descent(15)
	print "coordinate descent error w/ 15 iterations:", error(u, v)

################################################################################################################################################################################
'''
In this problem we choose a random line in the plane X = [-1, 1] x [-1, 1] as a decision boundary.
For points above the line we have the target f(x) = 1 and for points below the line we have f(x) = 0.
In this case f(x) represents a probability where y = +1 w/ prob f(x) and y = -1 w/ prob  1 - f(x)

We are going to use logistic regression with stochastic gradient descent to learn f.
'''
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def cross_entropy_error(w, x, y):
	x = np.insert(x, 0, 1, axis=1)
	if not (x.shape == (y.shape[0], 3) and w.shape[0] == 3):
		raise ValueError
	signal = x.dot(w)
	s = 1 / sigmoid(y * signal)
	return np.sum(np.log(s)) / x.shape[0]

# gradient is evaluated with weights w at single input x and output y
def cross_entropy_gradient(w, x, y):
	denom = 1 + np.exp(y * (w.dot(x)))
	num = y * x
	return -(num / denom)

class LogisticRegression(object):
	"""Logistic Regression w/ stochastic gradient descent"""
	def __init__(self, d, x, y):
		''' 
		Inputs:
			d: dimension of data
			x: matrix of d-dimensional points
			y: correct classification for x
		'''
		super(LogisticRegression, self).__init__()
		assert x.shape[1] == d
		self.N = x.shape[0]
		self.d = d
		self.y = y
		self.eta = 0.01

		# set x0 = 1 for each data point
		self.x = np.insert(x, 0, 1, axis=1)
		self.w = np.zeros(d + 1)
		self.run()

	# Stochastic Gradient Descent for Logistic Regression
	def perform_sgd(self):
		old_w = np.array([100, 100, 100])
		self._iter = 0
		# keep updating till weights stop changing by more than 0.01
		while np.linalg.norm(old_w - self.w) >= 0.01:
			old_w = np.copy(self.w)
			self._iter += 1
			# random permutation for order in which points	 are used for weight udpate
			perm = np.random.permutation(self.N)
			for i in perm:
				self.update_weights(self.x[i], self.y[i])
		return self._iter

	def update_weights(self, point, output):
		self.w = self.w - self.eta * cross_entropy_gradient(self.w, point, output)

	def apply_regression(self, x):
		assert x.shape[1] == self.d
		x = np.insert(x, 0, 1, axis=1)
		return sigmoid(x.dot(self.w))

	def run(self):
		return self.perform_sgd()

	def reset(self):
		self.w = np.zeros(self.d + 1)
		

def rand_point():
	return tuple(np.random.uniform(-1, 1, 2))

def generate_randline():
	p1 = rand_point()
	p2 = rand_point()
	while p2 == p1:
		p2 = rand_point()
	(x1, y1) = p1
	(x2, y2) = p2
	slope = (y2 - y1) / (x2 - x1)
	f = lambda x: ((x - x1) * slope) + y1 
	return f

def gen_data(f, N):
	x = []
	y = []
	# generate dataset of size N
	for i in xrange(N):
		p = rand_point()
		# find point on f corresponding to x value of p
		boundary = f(p[0])
		# classify accordingly if p above f or below f: cmp returns 1 if greater, 0 if equal, -1 if less
		y.append(cmp(p[1], boundary))
		x.append(p)
	return (np.array(x), np.array(y))

def logistic_regression_problem():
	N = 100
	d = 2
	f = generate_randline()
	x, y = gen_data(f, 100)
	lr = LogisticRegression(d, x, y)
	print "In sample error:", cross_entropy_error(w, x, y)

	test_x, test_y = gen_data(f, 1000)
	print "Out of sample error", cross_entropy_error(w, test_x, test_y)

if __name__ == '__main__':
	#grad_descent_problem()
	logistic_regression_problem()

import numpy as np

""" 
In this problem f is a random line representing the decision boundary. 
Points above this line are 1's and points below are -1's.
We are using the PLA to learn this decision boundary from already classified points.
"""
rand_point = lambda: tuple(np.random.uniform(-1, 1, 2))
def generate_f():
	p1 = rand_point()
	p2 = rand_point()
	while p2 == p1:
		p2 = rand_point()
	(x1, y1) = p1
	(x2, y2) = p2
	slope = (y2 - y1) / (x2 - x1)
	f = lambda x: ((x - x1) * slope) + y1 
	return f

def generate_data(f, N):
	x = []
	y = []
	for i in xrange(N):
		p_i = rand_point()
		e = f(p_i[0])
		if e <= p_i[1]:
			y.append(1)
		else:
			y.append(-1)
		x.append(p_i)
	return np.array(x), np.array(y)

def out_of_sample_err(correct, classified):
	if not correct.shape == classified.shape:
		raise ValueError
	num_samples = correct.shape[0]
	num_errors = np.count_nonzero(correct != classified)
	return float(num_errors) / num_samples

class PLA(object):
	"""Implementation of Perceptron Learning Algorithim""" 
	def __init__(self, d, x, y):
		''' 
		bias: bias/threshold 
		d: dimension of data
		x: input data matrix of d-dimensional points
		y: correct classification for x
		'''
		super(PLA, self).__init__()
		if not x.shape[1] == d:
			raise ValueError
		self.N = x.shape[0]
		self.d = d
		self.y = y
		# set x0 = 1 for each data point
		self.x = np.insert(x, 0, 1, axis=1)
		self.w = np.array([0 for i in xrange(d + 1)])

	def sign(self, x):
		if x == 0:
			return 0
		elif x < 0:
			return -1
		return 1
	
	def update(self):
		'''
		Runs one interation of PLA update
		Returns: True if no update was done, False otherwise.
		'''
		h = np.array([self.sign(self.w.dot(self.x[i])) for i in xrange(self.N)])

		# Runs update on first error
		for i in xrange(self.N):
			if h[i] != self.y[i]:
				self.w = self.w + (self.x[i] * self.y[i])
				return False
		return True

	def run(self):
		'''
		Return: number of iterations till convergence and final weights
		'''
		i = 1
		while not self.update():
			i += 1
		return i, self.w

	def reset(self):
		self.w = np.array([0 for i in xrange(self.d + 1)])

	def classify(self, x):
		'''
		Classify dataset
		Returns: Classification of 1 or -1 for each point
		'''
		if not x.shape[1] == d:
			raise ValueError
		N = x.shape[0]
		x = np.insert(x, 0, 1, axis=1)
		h = np.array([self.sign(self.w.dot(x[i])) for i in xrange(N)])
		return h



if __name__ == '__main__':
	d = 2
	N = 100
	total = 0
	for _ in xrange(1000):
		f = generate_f()
		x, y = generate_data(f, N)
		pla = PLA(d, x, y)
		_iter = pla.run()[0]	
		total += _iter
	print "average iterations till convergence:", float(total)/1000

	out_error = 0
	for _ in xrange(100):
		x, y = generate_data(f, 10000)
		classified = pla.classify(x)
		out_error += out_of_sample_err(y, classified)
	print "average out of sample error rate:", out_error / 100
	
import pdb # debugger
import random 
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

BOLD = '\033[1m'
END = '\033[0m'
	
# Restricts input to interval
def clamp(x, minx, maxx):
    if x < minx:
        return minx
    elif x > maxx:
        return maxx
    else:
        return x

'''
In this problem we use the following experiment:
Flip a 1000 coins 10 times each. Take c1 = first coin, crand = randomly chosen coin, cmin = coin with least heads
and record each coin's frequency of heads.

We run this experiment 10,000 times to get an average frequency of heads for each coin after 10,000 runs.
'''

def flip_n(n):
	# heads = 0, tails = 1
	c = [0, 1]
	return [random.choice(c) for _ in xrange(n)]

def run_sim():
	h_freq = lambda l: float(l.count(0)) / len(l)
	results = [flip_n(10) for _ in xrange(1000)]
	c1 = results[0]
	c_rand = random.choice(results)
	c_min = min(results, key=h_freq)
	# return [v1, v_rand, v_min] in that order
	return [h_freq(c) for c in (c1, c_rand, c_min)]

# run code for first problem
def coin_sim_problem():
	print "{0}Coin Simulation:{1}".format(BOLD, END)
	v1 = v_rand = v_min = 0.0
	num_sim = 1000
	# simulate 100,000 times
	sims = [run_sim() for i in xrange(num_sim)]
	i = 0
	for s in sims:
		v1 += s[0]
		v_rand += s[1]
		v_min += s[2]
		i += 1

	# calculate average values
	v1 = v1 / num_sim
	v_rand = v_rand / num_sim
	v_min = v_min / num_sim
	print(v1, v_rand, v_min)


'''
In this problem f is a random line representing the decision boundary. 
Points above this line are 1's and points below are -1's.
We are using linear regression for classification, from a training data set we generate.
'''
class LinearRegressionClassifier(object):
	"""
	Classifier using linear regression. 
	Uses sign(w * x) where w are the learned weights and x is an input
	for binary output of -1 or +1.
	"""
	def __init__(self, d, x, y):
		''' 
		Inputs:
			d: dimension of data
			x: matrix of d-dimensional points
			y: correct classification for x
		'''
		super(LinearRegressionClassifier, self).__init__()
		if not x.shape[1] == d:
			raise ValueError
			
		self.N = x.shape[0]
		self.d = d
		self.y = y

		# set x0 = 1 for each data point
		self.x = np.insert(x, 0, 1, axis=1)
		self.w = np.zeros((1, d + 1))
		self.run()

	def sign(self, x):
		if x == 0:
			return 0
		elif x < 0:
			return -1
		return 1

	def run(self):
		'''
		Solves and sets weights for linear regression on input data x.
		'''
		# pdb.set_trace()
		pseudo_inv = np.linalg.pinv(self.x)
		self.w = pseudo_inv.dot(self.y)

	def reset(self):
		'''
		Reset algorithim by setting all weights to 0
		'''
		self.w = np.zeros((1, d + 1))

	def get_weights(self):
		'''
		Returns: Current weights in self.w
		'''
		return self.w

	def classify(self, x):
		'''
		Classifies input dataset x.
		Returns: Classification of 1 or -1 for each point
		'''
		if not x.shape[1] == self.d:
			raise ValueError

		N = x.shape[0]
		x = np.insert(x, 0, 1, axis=1)
		return np.array([self.sign(self.w.dot(x[i])) for i in xrange(N)])


def rand_point():
	return tuple(np.random.uniform(-1, 1, 2))

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
	# generate dataset of size N
	for i in xrange(N):
		p_i = rand_point()
		# find point on f corresponding to x value of p_i
		e = f(p_i[0])
		# classify accordingly if p_i above f or below f: cmp returns 1 if greater, 0 if equal, -1 if less
		y.append(cmp(p_i[1], e))
		x.append(p_i)
	return (np.array(x), np.array(y))

def calc_error_rate(correct, classified):
	if not correct.shape == classified.shape:
		raise ValueError
	num_samples = correct.shape[0]
	num_errors = np.count_nonzero(correct != classified)
	return float(num_errors) / num_samples

# run code for second problem/first linear regression problem
def lin_reg1():
	print "{0}Lin Reg 1:{1}".format(BOLD, END)
	d = 2
	N = 100
	total = 0
	error_rate = 0.0
	num_sim = 100
	for _ in xrange(num_sim):
		f = generate_f()
		x, y = generate_data(f, N)
		lrc = LinearRegressionClassifier(d, x, y)
		classified = lrc.classify(x)
		error_rate += calc_error_rate(y, classified)
	print "Average in sample error:", error_rate / num_sim

	'''
	Plot f and the line learned through linear regression
	Note: The function we are trying to learn is the binary function where points above f = +1, and points below f = -1.
	Thus our function g that is learned is also a binary classifier. 
	We solve for x2 to find an equation based on single x values so we can plot the line. 
	'''  
	f = generate_f()
	x, y = generate_data(f, N)
	lrc_out = LinearRegressionClassifier(d, x, y)

	(w0, w1, w2) = lrc_out.get_weights()
	if w2 == 0:
		w2 = 1
	g = lambda x: (w0 + w1 * x) / -w2
	diff = lambda x: abs(clamp(f(x), -1, 1) - clamp(g(x), -1, 1))
	diff_area = integrate.quad(diff, -1, 1)[0] / 4.0
	print "Ratio of yellow area to graph area (this is the out of sample error rate):", diff_area

	x_points = np.linspace(-1, 1, 100)
	plt.xlim(-1, 1)
	plt.ylim(-1, 1)
	plt.plot(x_points, f(x_points), label='f - the target function', color='red')
	plt.plot(x_points, g(x_points), label='g - the learned function', color='blue')
	plt.legend()
	diff_poly = plt.fill_between(x_points, f(x_points), g(x_points), color='yellow', alpha = 0.4)
	plt.show()
	
	

if __name__ == '__main__':
	barrier = '-' * 20
	coin_sim_problem()
	print barrier
	lin_reg1()

import numpy as np

def target(x):
	return np.sin(np.pi * x)

def gen_data(n):
	samples = np.random.uniform(-1, 1, n)
	return np.array(zip(samples, target(samples)))

# create hypothesis of form g(x) = ax that minimizes MSE over two examples i.e linear regression through origin
def hypothesis(data):
	num = np.prod(data[0]) + np.prod(data[1])
	denom = sum([data[0][0]**2, data[1][0]**2])
	a_hat =  num / denom
	return a_hat

def calc_average_hypothesis(s):
	a = []
	for _ in xrange(s):
		d = gen_data(2)
		a.append(hypothesis(d))
	a = np.array(a)
	return (np.mean(a), a)

def calc_mse(g_hat, data):
	error = 0.0
	for d in data:
		error += pow(g_hat * d[0] - d[1], 2)
	return error / data.shape[0]

def calc_varx(g, g_hat, data):
	squared_diff = 0.0
	for d in data:
		squared_diff += pow(g * d[0] - g_hat * d[0], 2)
	return squared_diff

if __name__ == '__main__':
	num_hypotheses = 1000
	test_size = 1000
	g_hat, g_arr = calc_average_hypothesis(num_hypotheses)
	testing = gen_data(test_size)
	print "bias: ", calc_mse(g_hat, testing)

	t_varx = 0.0
	for g in g_arr:
		t_varx += calc_varx(g, g_hat, testing)
	varx = t_varx / num_hypotheses
	var = varx / test_size
	print "variance: ", var
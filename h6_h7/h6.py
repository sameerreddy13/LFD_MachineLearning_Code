import pdb
import numpy as np
from load_data import get_train, get_test
import sys
sys.path.append("..")
import h2

''' 
In this problem we perform linear regression on a set of labeled two dimensional inputs in R^2. 
We also apply a nonlinear transform (x1, x2) -> (x1, x2, x1^2, x2^2, x1*x2, |x1 - x2|, |x1 + x2|).
We compare our error with and without regularization
'''

def nonlinear_transform(x):
	f = lambda x: [x[0], x[1], x[0] ** 2, x[1] ** 2, x[0] * x[1], abs(x[0] - x[1]), abs(x[0] + x[1])]
	return np.array([f(x_i) for x_i in x])

if __name__ == '__main__':
	x_train, y_train = get_train()
	x_test, y_test = get_test()
	x_train = nonlinear_transform(x_train)
	x_test = nonlinear_transform(x_test)

	lrc = h2.LinearRegressionClassifier(7, x_train, y_train)
	in_sample_err = h2.calc_error_rate(y_train, lrc.classify(x_train))
	out_sample_err = h2.calc_error_rate(y_test, lrc.classify(x_test))
	h2.boldprint("Without Regularization:")
	print "In sample err", in_sample_err
	print "Out of sample err", out_sample_err

	_lambda = .5
	lrc = h2.LinearRegressionClassifier(7, x_train, y_train, reg=True, reg_param=_lambda)
	in_sample_err = h2.calc_error_rate(y_train, lrc.classify(x_train))
	out_sample_err = h2.calc_error_rate(y_test, lrc.classify(x_test))
	h2.boldprint("With Regularization:")
	print "In sample err", in_sample_err
	print "Out of sample err", out_sample_err
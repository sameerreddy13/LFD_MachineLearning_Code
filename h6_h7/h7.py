### Incomplete ###
import pdb
import numpy as np
from load_data import get_train_and_validation, get_test
import sys
sys.path.append("..")
import h2

def nonlinear_transform(x):
	f = lambda x: [x[0], x[1], x[0] ** 2, x[1] ** 2, x[0] * x[1], abs(x[0] - x[1]), abs(x[0] + x[1])]
	return np.array([f(x_i) for x_i in x])

if __name__ == '__main__':
	x_train, y_train, x_valid, y_valid = get_train_and_validation()
	x_test, y_test = get_test()
	x_train = nonlinear_transform(x_train)
	x_test = nonlinear_transform(x_test)
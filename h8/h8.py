import pdb
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV

import sys
sys.path.append("..")
from h2 import calc_error_rate, boldprint

###################### Load Data ######################
'''
We first have functions to load handwritten digit data from the US Postal Service Zip Code data set. 
Each digit has its label, intensity, and symmetry. Thus we have 3 dimensional data points.
'''

def get_train():
	columns = ['digits', 'intensity', 'symmetry']
	train = pd.read_csv("train.txt", sep="  ", header=None, names=columns, engine='python')
	return train

def get_test():
	columns = ['digits', 'intensity', 'symmetry']
	test = pd.read_csv("test.txt", sep="  ", header=None, names=columns, engine='python')
	return test

###################### Train and Test ######################
'''
We are going to use a soft-margin SVM to train two types of binary classifiers:
A one-versus-one (one digit is +1, one is -1, the rest are disregarded) and
A one-versus-all (one digit is +1, rest are -1)
'''
def create_labels(positive=None, negative='all', digits=None):
	'''
	Create the y labels for dataset x.
	Parameters:
		positive: define what digit should be +1
		negative: define what digit should be -1. If doing one-versus-all don't define this parameter.
		digits: array of digits. 

	Returns: 
	numpy array of +1, -1 and 0 (disregarded digits in one-vs-one).
	'''
	y = []
	assert positive is not None and negative is not None and digits is not None
	for d in digits:
		if d == positive:
			y.append(1)
		elif (negative == 'all' or d == negative):
			y.append(-1)
		else:
			y.append(0)
	return np.array(y)

get_x = lambda df: np.array(zip(df.intensity, df.symmetry))

def one_v_all_label(d, df):
	return np.array([1 if num == d else -1 for num in df.digits])

def one_v_one_label(pos, neg, df):
	y = []
	for num in df.digits:
		if num == pos:
			y.append(1)
		elif num == neg:
			y.append(-1)
	return np.array(y)

if __name__ == '__main__':
	train = get_train()
	test = get_test()
	dig = np.arange(0, 10)
	models = []
	boldprint("One-vs-all classifiers")
	print "Digit", "\tIn-Sample Err", "\tOut-Sample Err"
	for d in dig:
		y = one_v_all_label(d, train)
		x = get_x(train)
		model = svm.SVC(C=0.01, kernel="poly", degree=2, decision_function_shape='ovr', gamma=1, coef0=1) 
		model.fit(x, y)
		models.append(model)
		print d, "\t", round(1 - model.score(x, y), 4), "\t\t", round(1 - model.score(get_x(test), one_v_all_label(d, test)), 4)

	boldprint("Making 1 v 5 classifiers...")
	train_1_5 = train[train.digits.isin([1, 5])]
	test_1_5 = test[test.digits.isin([1, 5])]
	y = one_v_one_label(1, 5, train_1_5)
	x = get_x(train_1_5)
	for C in [0.001, 0.01, 0.1, 1]:
		model = svm.SVC(C=C, kernel="poly", degree=2, decision_function_shape='ovr', gamma=1, coef0=1) 
		model.fit(x, y)
		print "C =", C
		print "Ein =", 1 - model.score(x, y)
		print "Eout =", 1 - model.score(get_x(test_1_5), one_v_one_label(1, 5, test_1_5))
		print "Num Support Vectors =", model.n_support_
		print ""

	boldprint("Cross validation...")
	model = GridSearchCV(svm.SVC(kernel='poly', degree=2, gamma=1, coef0=1, decision_function_shape='ovr'), [{'C' : [0.0001, 0.001, 0.01, 0.1, 1]}], cv=10)
	model.fit(x, y)
	print model.best_params_
	print 1 - model.score(x, y)
	print 1 - model.score(get_x(test_1_5), one_v_one_label(1, 5, test_1_5))



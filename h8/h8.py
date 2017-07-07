import pdb
import pandas as pd
import numpy as np
from sklearn import svm

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
	@param positive: define what digit should be +1
	@param negative: define what digit should be -1. If doing one-versus-all don't define this parameter.
	@param digits: array of digits. 

	Returns: 
	numpy array of +1, -1 and 0 (disregarded digits in one-vs-one). Label order corresponds to digits input order.
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

def error_wrt_label(label, df, m):
	assert label is not None
	df = df[df.digits == label]
	predictions = m.predict(get_x(df))
	print predictions
	return len(np.where(x != label)) / float(len(predictions))

if __name__ == '__main__':
	train = get_train()
	test = get_test()
	dig = np.arange(0, 10)
	models = []
	boldprint("One-vs-all classifiers")
	y = np.array(train.digits)
	x = get_x(train)
	model = svm.SVC(C=0.01, kernel="poly", degree=2, decision_function_shape='ovr') 
	model.fit(x, y)
	# for d in dig:
	# 	print d, error_wrt_label(d, train, model)
	print error_wrt_label(1, train, model)
	print model.n_support_
	# out_sample_err = calc_error_rate(create_labels(positive=d, digits=test.digit), model.predict(get_x(test)))
	# in_sample_err = calc_error_rate(y, model.predict(x))
	# print "Digit | In Sample Err | Out Sample Err" 
	# print 0, '\t', round(in_sample_err, 4), '\t\t', round(out_sample_err, 4)


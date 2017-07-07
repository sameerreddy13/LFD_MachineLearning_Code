import numpy as np
import pandas as pd

def transform_to_numpy(df):
	x = np.array(zip(df.x1, df.x2))
	y = np.array(df.y)
	return x, y

def get_train():
	# Each row corresponds to a two dimensional point x = (x1, x2) with the third column y = -1 or +1
	columns = ['x1', 'x2', 'y']
	train = pd.read_csv("train.dta.txt", sep="  ", header=None, names=columns, engine='python')
	x_train, y_train = transform_to_numpy(train)
	return x_train, y_train

def get_test():
	# Each row corresponds to a two dimensional point x = (x1, x2) with the third column y = -1 or +1
	columns = ['x1', 'x2', 'y']
	test = pd.read_csv("test.dta.txt", sep="  ", header=None, names=columns, engine='python')
	x_test, y_test = transform_to_numpy(test)
	return x_test, y_test
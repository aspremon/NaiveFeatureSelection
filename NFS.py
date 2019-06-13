import numpy as np
from numba import jit
#import math
#import time
#import pdb
#import sys


def split_classes(X,y,label=0):
	"""
	split_classes function

	Description: Function that takes the original
	data and partitions it into two classes.

	Arguments:
		- X: training data as a n x m matrix (n datapoints
		with m features per data point)
		- Y: class labels for the n different points
		- label: desired class

	Returns:
		- x1: all the data for class = label
		- x2: the data for class != label

	Note: this only works for binary classification
	"""
	idx1 = np.where(y == label)[0]
	idx2 = np.where(y != label)[0]
	return {'class1': X[idx1,:], 'class2': X[idx2,:]}


def class_prob(y,label = 0):
	"""
	class_prob function

	Description: Function that finds the ratio
	of class1 and class2 labels to use as the
	prior probability for naive bayes classification.

	Arguments:
		- Y: class labels for the n different points
		- label: desired class
	
	Returns:
		- pc1: probability of class = label
		- pc2: probability of class != label
	"""
	n = len(y)
	idx = np.where(y == label)[0]
	return {'pc1': len(idx)*1.0/n, 'pc2': (n-len(idx))*1.0/n}


#@jit(nopython=True)
def entr(x):
	"""
	Negative entropy function.

	This returns sum_i x_i * log(x_i) 
	"""
	res = np.zeros(x.shape)
	res[np.nonzero(x)] = x[np.nonzero(x)] * np.log(x[np.nonzero(x)])
	return res


#@jit(nopython=True)
def fun(a,k,c,f1,f2):
	"""
	Auxialiary gradient function.
	"""
	h = c - f1 * np.log(a) - f2 * np.log(1 - a)
	idx = np.argpartition(h, -k)[-k:]
	#fval = sum(h[idx])
	nabla_a = -f1 / a + f2 / (1 - a)
	df = np.sum(nabla_a[idx])
	return df


#@jit(nopython=True)
def nfs(X,y,k):
	"""
	Naive Feature Selection

	Description: Function that solves the following multinomial naive Bayes
	training problem, written

	max_{q,r}   fp^T log q + fm^T log r

		s.t. 	\|q - r\|_1 <= k
				0 <= q, r
			 	sum(q) = 1, sum(r) = 1

	Arguments:
		- X: training data as a n x m matrix (n datapoints
		with m features per data point)
		- Y: binary class labels for the n different points (one of the labels should be 1)
		- k: target cardinality for vector in linear classification rule

	Returns:
		- idx: 	Index of selected features
		- q,r: 	Solutions reconstructed from optimal dual solution
		- w0,w: linear classification rule: w0 + w'*x > 0, where w is k sparse.
	"""

	# First construct fp, fm, by summing feature vectors for each class
	#m = X.shape[1]
	#n = X.shape[0]
	split = split_classes(X,y,label=1)
	C1 = split['class1'].shape[0]
	C2 = split['class2'].shape[0]
	f1 = np.sum(split['class1'],axis=0) 
	f2 = np.sum(split['class2'],axis=0)
	f1 = np.squeeze(np.asarray(f1))
	f2 = np.squeeze(np.asarray(f2))

	# Define dual objective function
	alpha_low = 0
	alpha_top = 1
	c = entr(f1) + entr(f2) - entr(f1+f2)
	# Solve dual problem by bisection
	tol = 1e-6
	while (alpha_top - alpha_low) > 1e-10:
		alpha = (alpha_top + alpha_low) / 2.0
		df = fun(alpha,k,c,f1,f2)
		# print(df)
		if df > tol:
			alpha_top = alpha
		elif df < -tol: # TODO: check if OK stopping on gradient?
			alpha_low = alpha
		else:
			break

	# Get primal points from dual solution
	h = c - f1 * np.log(alpha) - f2 * np.log(1 - alpha)
	idx = np.argpartition(h, -k)[-k:]
	mask = np.ones(len(f1), dtype=bool)  # all elements included/True.
	mask[idx] = False
	qopt = np.zeros(len(f1))
	ropt = np.zeros(len(f2))
	qopt[mask] = (f1[mask] + f2[mask]) / sum(f1 + f2)
	ropt[mask] = qopt[mask]
	qopt[idx] = sum(f1[idx] + f2[idx]) / (sum(f1[idx]) * sum(f1 + f2)) * f1[idx]
	ropt[idx] = sum(f1[idx] + f2[idx]) / (sum(f2[idx]) * sum(f1 + f2)) * f2[idx]

	# Get linear classification rule: w0 + w'*x > 0
	pc = class_prob(y,label=1)
	w0 = np.log(pc['pc1']) - np.log(pc['pc2'])
	with np.errstate(divide='ignore'):
		w1 = np.log(qopt)
		idx_inf = np.isinf(w1)
		w1[idx_inf] = -33.6648265 #essentially the 0 entries, log(0) to -33.6648
		w2 = np.log(ropt)
		idx_inf = np.isinf(w2)
		w2[idx_inf] = -33.6648265
	w = w1 - w2

	# Return results
	return {'idx' : idx,'w0' : w0, 'w' : w,'q' : qopt, 'r' : ropt}

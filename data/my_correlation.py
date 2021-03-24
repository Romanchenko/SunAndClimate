import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import math


def self_corr(x):
	x1 = x[:-1]
	x2 = x[1:]
	n1 = len(x1)
	n2 = len(x2)
	return sum((x1 - np.mean(x1) * (x2 - np.mean(x2)))) / (n1 * n2)


def variance(x):
	x_mean = np.mean(x)
	return sum((x - x_mean)**2) / (len(x)**2)


def corr_coeff(x1, x2):
	n = len(x1)
	numerator = sum((x1 - np.mean(x1)) * (x2 - np.mean(x2)))
	v1 = sum((x1 - np.mean(x1))**2) / len(x1)
	v2 = sum((x2 - np.mean(x2))**2) / len(x2)
	denominator = v1 * v2
	return numerator / (np.sqrt(denominator) * n)


def test_sample(x, y):
	actual = corr_coeff(x, x)
	expected = np.corrcoef(np.array([x, y]))[0][1]
	eps = 10**(-6)
	if abs(actual - expected) > eps:
		print('ACHTUNG: expected ', expected, ', but actual is', actual, 'sample is: x =', x, ' y =', y)
		exit(0)


def test():
	test_sample(x=[1, 2, 3], y=[2, 3, 4])
	test_sample(x=[2, 1, 0], y=[0, 1, 2])
	test_sample(x=[-2.1, -1,  4.3], y=[3,  1.1,  0.12])
	print('all tests passed')


def __main__():
	test()

import numpy as np
import matplotlib.pyplot as plt
import sys
import statsmodels.api as sm
import pandas as pd
import math
import my_correlation
import datetime
import random

np.seterr(divide='ignore', invalid='ignore')

def autocorr(x, t=1):
	if 'date_in_days' not in x.columns:
		make_dates(x)
	x1 = x.iloc[t:]
	x2 = x.iloc[:-t]
	x1 = x1.reset_index(drop=True)
	x2 = x2.reset_index(drop=True)
	good_points = (x1['date_in_days'] == x2['date_in_days'] + t)
	x1 = x1[good_points]
	x2 = x2[good_points]
	corr_c = np.corrcoef(np.array([x1.where(good_points)['val'], x2.where(good_points)['val']]))[0][1]
	return corr_c

def plot_autocorrelation(x, window, to_plot=True):
	sz = x.shape[0]
	print(sz)
	vals = [0] * sz
	for i in range(window - 1, sz):
		vals[i] = autocorr(x.iloc[i - window + 1:i + 1])
	x['corrcoef'] = vals
	if to_plot:
		plt.plot(x['year'], x['corrcoef'])

		
def plot_autocorrelation_all(dict_x, window, names=['1', '2', '3', '4'], fname=None):
	
	plt.figure(figsize=(12, 7))
	plt.grid(True)
	for name in names:
		x = dict_x[name]
		sz = x.shape[0]
		x = x.reset_index()
		vals = [0] * sz
		progress = 0
		# [pos - window // 2, pos + (window + 1) // 2]
		for i in range(window // 2, sz - (window + 1) // 2 + 1):
			vals[i] = autocorr(x.loc[i - window // 2 : i + (window + 1) // 2])
			
		print('Finished step for ', name, ' in ', ('None' if (fname == None) else fname))
		x['corrcoef'] = vals
		first_year = x['year'][0]
		last_year = x['year'][sz - 1]
		plt.plot(np.linspace(first_year, last_year, len(x['corrcoef'])), x['corrcoef'], label=name)
	plt.legend()
	if fname != None:
		plt.savefig(fname)
		print('saved result as:', fname)
	else:
		plt.show()


months = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
months_cumsum = np.cumsum(months)


def get_month(x):
	f, i = math.modf(x)
	f *= 365
	f = int(f)
	for i in range(1, 13):
		if months_cumsum[i - 1] <= f <= months_cumsum[i]:
			return i
	print('Error! Can\'t find the month for: ', x)

# df should have cols: year, month, day
def make_dates(df):
	transformer = lambda dt: datetime.date(int(dt['year']), int(dt['month']), int(dt['day'])).toordinal() 
	df['date_in_days'] = df.apply(transformer, axis=1)


def work(data, window=1000, name_list=['winter', 'spring', 'summer', 'autumn', 'total'], fname=None):
	if data.shape[1] != 4:
		data.columns = ['year', 'val']
		data['month'] = list(map(get_month, data['year']))
		# TODO - make days here
	else:
		data.columns = ['year', 'month', 'day', 'val']
	seasons = {}
	make_dates(data)
	seasons['summer'] = data.loc[(data['month'] <= 8) & (data['month'] >= 6)]
	seasons['winter'] = data.loc[(data['month'] == 12) | (data['month'] >= 1) & (data['month'] <= 2)]
	seasons['autumn'] = data.loc[(9 <= data['month']) & (data['month'] <= 11)]
	seasons['spring'] = data.loc[(3 <= data['month']) & (data['month'] <= 5)]
	seasons['total'] = data
	N = data.shape[0]
	plot_autocorrelation_all(seasons, window=window, names=name_list, fname=fname)	


def test_data_transform():
	data_dict = {'year': [2000, 2000, 2001, 2002], 'month': [8, 8, 8, 8], 'day': [1, 2, 1, 1]}
	df = pd.DataFrame.from_dict(data_dict)
	make_dates(df)
	days = df['date_in_days']
	if days[1] - days[0] != 1 or days[2] - days[0] != 365 or days[3] - days[2] != 365:
		print('Error in date transform')
		exit(0)
	print('Date transform works')

def test_autocorr_with_date_correction():
	data_dict = {
		'year': [2020, 2020, 2020, 2020, 2021, 2021, 2021],
		'month': [5, 5, 5, 5, 3, 3, 3],
		'day': [27, 28, 30, 31, 1, 2, 3],
		'val': [1, 2, 1, 2, 30, 20, 30]
	}
	df = pd.DataFrame.from_dict(data_dict)
	# assuming arrays to be:
	# [1, 2, 1, 2 , 30, 20] -> [1, 2, 1, 30, 20]
	# [2, 1, 2, 30, 20, 30] -> [2, 1, 2, 20, 30]
	# mean_1 = 10.8
	# mean_2 = 11
	# substracted means: [-9.8, -8.8, -9.8, 19.2,  9.2], [ -9, -10,  -9,   9,  19]
	# dot product: 612
	# std_1 = 12.023310692151309
	# std_2 = 11.865917579353061
	# expected = 0.8579382301433244
	# problem with means - they are actually different. 
	# Well, let's postpone this question for a bit and see how this exact thing will work =)
	actual = autocorr(df)
	expected = 0.8579382301433244
	eps = 1**(-5)
	if abs(actual - expected) > eps:
		print('ACHTUNG! Autocorrelation was calculated with errors. Actual is', actual, ', expected is', expected)
		exit(0)
	print('Autocorrelation considering time gaps is calculated correctly')


def test():
	test_data_transform()
	test_autocorr_with_date_correction()

test()
data = pd.read_csv(sys.argv[1], sep='\s+')
window = 1000
if len(sys.argv) > 2:
	window = int(sys.argv[2])
names = ['winter', 'spring', 'summer', 'autumn', 'total']
if len(sys.argv) > 3:
	names = sys.argv[3].split()
fname = None
if len(sys.argv) > 4:
	fname = sys.argv[4]
work(data, window, names, fname)

#data = pd.read_csv(sys.argv[1], sep='\s+', names=['year', 'month', 'day', 'val'])
#data['month'] = list(map(int, data['month']))

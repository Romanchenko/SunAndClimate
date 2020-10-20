import numpy as np
import math
from scipy.stats import norm
def get_binary_map(arr, threshold):
    return np.array(arr) >= threshold


def get_entropy(arr):
    counter = {}
    for bin_str in arr:
        if bin_str in counter:
            counter[bin_str] += 1
        else:
            counter[bin_str] = 1
    entr = 0
    for k, v in counter.items():
        prob = v / len(arr)
        entr -= math.log(prob, 2) * prob
    return entr


# get S(p, a) = E(2) - E(1)
def get_entropy_diff(s):
    counter = {}
    counter1 = [0, 0]
    n = len(s)
    for i in range(0, n - 1):
        if s[i:i+2] not in counter:
            counter[s[i:i+2]] = 1
        else:
            counter[s[i:i+2]] += 1
        counter1[int(s[i])] += 1
    counter1[int(s[n - 1])] += 1
    entr = 0
    total_prob=0
    for k, v in counter.items():
        prob = v * 1.0 / (n - 1)
        total_prob += prob
        entr -= math.log(prob, 2) * prob
    for v in counter1:
        prob = v * 1.0 / n
        if prob > 0:
        	entr += math.log(prob, 2) * prob
    return entr


def retrive_p_from_data(data, threshold):
	data = np.array(data)
	mean_ = data.mean()
	std_ = data.std()
	return 1 - norm.cdf(threshold, loc=mean_, scale=std_)


def get_slope_entropy(p, k=100):
    eps = np.linspace(0.01, 1, k)
    slope = []
    for e in eps:
        s = p * (1 - p) * e * (2 * math.log(e, 2) + math.log(p, 2) + math.log(1-p, 2))
        s += p * (1 - e + p * e) * math.log(1 - e + p * e, 2)
        s += (1 - p) * (1 - p * e) * math.log(1 - p * e, 2)
        slope.append(-s)
    return eps, slope

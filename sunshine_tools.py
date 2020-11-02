import numpy as np
import math
from scipy.stats import norm
from collections import defaultdict

def get_binary_map(arr, threshold):
    return np.array(arr) >= threshold

def get_entropy_from_counters(counter_1, counter_2, window_size):
	entr = 0
    for k, v in counter_2.items():
        prob = v * 1.0 / (window_size - 1)
        if prob > 0:
        	entr -= math.log(prob, 2) * prob
    for v in counter_1.items():
        prob = v * 1.0 / window_size
        if prob > 0:
        	entr += math.log(prob, 2) * prob
    return entr


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
    for k, v in counter.items():
        prob = v * 1.0 / (n - 1)
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


def get_exper_entropy(data, threshold, window):
	data = np.array(data)
	bin_map = get_binary_map(data, threshold)
    n = len(data)
    res = []
    counter_1 = defaultdict(int)
    counter_2 = defaultdict(int)
    for i in range(window - 1):
    	counter_1[bin_map[i]] += 1
    	counter_2[str(bin_map[i]) + str(bin_map[i + 1])] += 1

    for i in range(window - 1, n):
    	counter_1[bin_map[i]] += 1
    	counter_2[str(bin_map[i]) + str(bin_map[i + 1])] += 1
    	if i >= window:
    		counter_1[bin_map[i - window]] -= 1
    		counter_2[str(bin_map[i - window]) + str(bin_map[i - window + 1])] -= 1	
        e = get_entropy_from_counters(counter_1, counter_2, window_size)
        res.append(e)
    return res


def find_lifetime(entropy, entropys_by_p):
    ans = 0
    idx = bisect.bisect(entropys_by_p[1], entropy)
    eps = entropys_by_p[0][idx]
    return 1/eps

# returns array of size data.size() - window + 1
def lifetimes_from_data(data, window, threshold):
	bin_map = get_binary_map(data, threshold)
	p = retrive_p_from_data(data)
	entropys_by_p = get_slope_entropy(p)
	entropys = get_exper_entropy(data, threshold, window)
	lifetimes = [find_lifetime(entropy, entropys_by_p) for entropy in entropys]
	return lifetimes

import numpy as np
import math
from scipy.stats import norm
import bisect
from collections import defaultdict

# group of methods for derivatives
def build_averaged_values_arr(arr, window_size):
    n = len(arr)
    av = np.zeros(n)
    cur_sum = sum(arr[:window_size - 1])
    for i in range(window_size - 1, n):
        cur_sum += arr[i]
        if i - window_size >= 0:
            cur_sum -= arr[i - window_size]
        av[i] = cur_sum / window_size
    return av

def build_averaged_values(df, window_size):
    n = df.shape[0]
    av = np.zeros(n)
    cur_sum = sum(df.loc[:window_size - 2, 'val'])
    for i in range(window_size - 1, n):
        cur_sum += df.loc[i, 'val']
        if i - window_size >= 0:
            cur_sum -= df.loc[i - window_size, 'val']
        av[i] = cur_sum / window_size
    return av

def build_derivatives(df):
    n = df.shape[0]
    deriv = np.zeros(n)
    for i in range(1, n):
        if 'year' not in df or df.loc[i, 'year'] == df.loc[i - 1, 'year']:
            deriv[i] = df.loc[i, 'val'] - df.loc[i - 1, 'val']
    return deriv


def build_relation(df, window_size):
    averaged_values = build_averaged_values(df, window_size)[window_size:]
    derivatives = np.abs(build_derivatives(df))[1:]
    average_derivatives = build_averaged_values_arr(derivatives, window_size)[window_size - 1:]
    n = df.shape[0]
    result = np.zeros(n)
    result[window_size:] = averaged_values / average_derivatives
    return result


def make_dates(df):
    transformer = lambda dt: datetime.date(int(dt['year']), int(dt['month']), int(dt['day'])).toordinal() 
    df['date_in_days'] = df.apply(transformer, axis=1)

def plot_relation(df, window_size):
    df['relation'] = build_relation(df, window_size)
    plt.figure(figsize=(12, 7))
    plt.grid(True)
    s = r'$Y(t) = \frac{X_T(t)}{\dot{X}_T(t)}$'
    s += " window=" + "{:.2f}".format(window_size/365) + " years"
    plt.title(s, fontsize=18)
    plt.xlabel('Time')
    plt.ylabel('Relation')
    n = df.shape[0]
    years = np.linspace(df.loc[window_size, 'year'], df.loc[n - 1, 'year'], n - window_size)
    plt.plot(years, df.loc[window_size:, 'relation'])
  
def plot_average(df, window_size):
    average = build_averaged_values(df, window_size)
    plt.figure(figsize=(12, 7))
    plt.grid(True)
    plt.title('av(X) window=' + "{:.2f}".format(window_size/365) + " years")
    plt.xlabel('Time')
    plt.ylabel('Average sunshine (hours)')
    n = df.shape[0]
    years = np.linspace(df.loc[window_size - 1, 'year'], df.loc[n - 1, 'year'], n - window_size + 1)
    plt.plot(years, average[window_size-1:])

 
def plot_deriv(df, window_size):
    deriv = abs(build_derivatives(df, window_size))
    plt.figure(figsize=(12, 7))
    plt.grid(True)
    plt.title('|av\'(X)| window=' + "{:.2f}".format(window_size/365) + " years")
    plt.xlabel('Time')
    plt.ylabel('Abs of derivative')
    n = df.shape[0]
    years = np.linspace(df.loc[window_size, 'year'], df.loc[n - 1, 'year'], n - window_size)
    plt.plot(years, deriv[window_size:])


# group of methods for MRC extraction

def get_binary_map(arr, threshold):
    return np.array(arr) >= threshold

def get_entropy_from_counters(counter_1, counter_2, window_size):
    entr = 0
    for k, v in counter_2.items():
        prob = v * 1.0 / (window_size - 1)
        if prob > 0:
            entr -= math.log(prob, 2) * prob
    for k, v in counter_1.items():
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
    bin_map = get_binary_map(data, threshold).astype(np.int)
    n = len(data)
    res = []
    counter_1 = defaultdict(int)
    counter_2 = defaultdict(int)
    for i in range(window - 1):
        counter_1[str(bin_map[i])] += 1
        if i > 0:
            counter_2[str(bin_map[i - 1]) + str(bin_map[i])] += 1
    for i in range(window - 1, n):
        counter_1[str(bin_map[i])] += 1
        counter_2[str(bin_map[i - 1]) + str(bin_map[i])] += 1
        if i >= window:
            counter_1[str(bin_map[i - window])] -= 1 
            counter_2[str(bin_map[i - window]) + str(bin_map[i - window + 1])] -= 1	
        e = get_entropy_from_counters(counter_1, counter_2, window)
        res.append(e)
    return res


def find_lifetime(entropy, entropys_by_p, window):
    ans = 0
    #print(entropys_by_p[1])
    idx = bisect.bisect(entropys_by_p[1], entropy )
    if idx == len(entropys_by_p[1]):
        print('ERROR: Can`t find epsilon for current entropy {:.5f}'.format(entropy))
        idx = len(entropys_by_p[1]) - 1
    eps = entropys_by_p[0][idx]
    return 1/eps

# returns array of size data.size() - window + 1
def lifetimes_from_data(data, window, threshold):
    bin_map = get_binary_map(data, threshold)
    p = retrive_p_from_data(data, threshold)
    #print('p = ', p)
    p = 0.5
    entropys_by_p = get_slope_entropy(p)
    entropys = get_exper_entropy(data, threshold, window)
    #print('Entropys: ', entropys_by_p)
    #print('From data: ', entropys)
    lifetimes = [find_lifetime(entropy, entropys_by_p, window) for entropy in entropys]
    return lifetimes



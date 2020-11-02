import sunshine_tools
import numpy as np
import math

def eq(x, y):
    eps = 10**(-5)
    return abs(x - y) < eps
  

def __test_get_binary_map():
    arr = [1, 2, 3, 4, 5, 6]
    bins = sunshine_tools.get_binary_map(arr, 3.5)
    expected = [False, False, False, True, True, True]
    for i in range(6):
        assert bins[i] == expected[i], "invalid binary_map i=" + str(i) + " expected=" + str(expected[i]) + " actual=" + str(bins[i])


def __test_get_entropy_diff():
    s = "01110"
    t = "10001"
    entr = sunshine_tools.get_entropy_diff(s)
    # -1/4 ln(1/4) * 2 - 1/2 * ln(1/2) = ln 2 + 1/2 ln(2)
    # -2/5 ln(2/5) - 3/5 ln(3/5) = -2/5 ln(2) + 2/5 ln(5) - 3/5 ln(3) + 3/5 ln(5)
    expected = 1.5 * math.log(2, 2) - (math.log(5, 2) - 0.4 * math.log(2, 2) - 0.6 * math.log(3, 2))
    assert eq(entr, expected), "expected=" + str(expected) + " actual=" + str(entr)

    entr2 =sunshine_tools.get_entropy_diff(t)
    assert eq(entr, expected)


def __test_retrive_p():
	data = [0.1, 0.2, 1, 2, 0.5]
	data = np.array([-1, 0, 1, 2, 3, 4, 5])
	assert eq(0.5, sunshine_tools.retrive_p_from_data(data, 2))


if __name__ == '__main__':
	__test_get_entropy_diff()
	__test_get_binary_map()
	__test_retrive_p()
# imports
import numpy as np
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import math
import pandas as pd

# set random generator
generator = np.random.RandomState(42)

# settings of graphics
width = 17
height = 5
matplotlib.rcParams['figure.figsize'] = [width, height]
matplotlib.rcParams['lines.markersize'] = 0.5
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Tahoma']

# matplotlib.rcParams['scatter.edgecolors'] = "black"

#  days of spring and autumn equinoxes
spring = 79
autumn = 266
DAYS_PER_YEAR = 365


class Generative:
    def __init__(self, my_generator, seed=42):
        self.my_generator = my_generator
        self.step = 0
        self.cur_val = -20
        self.randomness = np.random.RandomState(seed)
        self.seed = seed

    def next(self):
        self.step += 1
        nxt = self.my_generator(self.step, self.cur_val, self.randomness)
        self.cur_val = nxt
        return nxt

    def reset(self):
        self.randomness = np.random.RandomState(self.seed)
        self.step = 0
        self.cur_val = 0


def ar(p, base_value, noise_maker, to_add=10000):
    """
    Генератор последовательности по авторегрессионной модели
    Parameters:
        p : double
            Параметр памяти авторегрессионной модели.  p in [0,1]
        base_value : double
            Первое значение, с которого начинается генерация
        noise_maker : function
            Функция, возвращающая случайное значение - шум
        to_add : int
            Сколько элементов должно быть сгенерировано
    Returns:
        Массив(list) сгенерированных точек
    """

    new_values = np.array(base_value)
    new_values = np.append(new_values, [0] * to_add)
    for i in range(to_add):
        noise = noise_maker(i)
        new_values[i + 1] = p * new_values[i] + noise
    return new_values


def mFFT(arr, draw=False, name='a[n]', x1=0, x2=-1, y2=-1, norm=-1, draw_less=False, return_pair=False):
    """
    Parameters:
        arr : list
            Массив, для которого считается Фурье
        draw : bool
            Рисовать ли график
        name : str, optional
            Заголовок графика
        x1 : double, optional
            Левая граница по x на графике спектра
        x2 : double, optional
            Правая граница по x на графике спектра
        y2 : double, optional
            Верхняя граница по y на графике спектра
        norm : double
            Число для нормировки, стандартное - длина массива
        draw_less : bool
             Надо ли рисовать график последовательности вдобавок к спектру частот
        return_pair : bool
            Возвращать ли массив периодов (координаты по x)

    Returns:
        Если не выставлен флаг energy, то возвращается массив абсолютных значений амплитуд спектра.
        Если флаг выставлен, то возращается кортеж из четырех элементов - энергия амплитуды 27-дневного сигнала,
        энергия амплитуды полугодового сигнала, их отношение и массив абсолютных значений амплитуд сигнала.
    """
    if x2 == -1:
        x2 = len(arr) // 2
    if norm == -1:
        norm = len(arr)

    A = np.fft.rfft((arr - np.mean(arr)) / norm)
    n = np.arange(len(arr))
    n1 = len(arr) / n[1:]

    if draw:

        plt.rcParams['axes.grid'] = True
        if not draw_less:
            fig, ax = plt.subplots(2, figsize=(6, 4), dpi=150)
            plt.tight_layout()

            plt.subplots_adjust(hspace=0.5)
            ax[0].plot(n, arr, '.-')
            ax[0].set_title(name)

            ax[1].set_title('$A$')
            if y2 > 0:
                ax[1].set_ylim(0, y2)
            ax[1].set_xlim(x1, x2)
            ax[1].plot(n1[0: (len(arr) // 2 + 1)], np.abs(A), '-')
            ax[1].set_xlabel('Период (в днях)')
        else:
            fig, ax = plt.subplots(1, figsize=(4, 2), dpi=150)
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.5)
            ax.set_title('$A$')

            if y2 > 0:
                ax.set_ylim(0, y2)
            ax.set_xlim(x1, x2)
            ax.plot(n1[0: (len(arr) // 2 + 1)], np.abs(A), '-')
            ax.set_xlabel('Период (в днях)')

        plt.show()

    if return_pair:
        return np.abs(A), n1[0: (len(arr) // 2 + 1)]
    return np.abs(A)


def mFFTe(arr, norm=-1, ep1=29, radius1=3, ep2=183, radius2=2, silent=True):
    """
    Parameters:
        arr : list
            Массив, для которого считается Фурье и энергия в тояках ep1 и ep2
        norm : double
            Число для нормировки, стандартное - длина массива
        ep1: double
            Точка для поиска энергии (подразумеваем 27 дней)
        ep2: double
            Точка для поиска энергии (подразумеваем полгода)
        radius1 : double
            Радиус подсчета суммарной энергии для ep1
        radius2 : double
            Радиус подсчета суммарной энергии для ep2
        silent : bool
            Нужен ли отладочный вывод
        
    Returns:
        Если не выставлен флаг energy, то возвращается массив абсолютных значений амплитуд спектра.
        Если флаг выставлен, то возращается кортеж из четырех элементов - энергия амплитуды 27-дневного сигнала,
        энергия амплитуды полугодового сигнала, их отношение и массив абсолютных значений амплитуд сигнала.
    """
    if norm == -1:
        norm = len(arr)

    normed_amps = np.fft.rfft((arr - np.mean(arr)) / norm)
    n = np.arange(len(arr))
    n1 = len(arr) / n[1:]

    sum_amps = np.abs(normed_amps[(np.abs(n1[0: (len(arr) // 2 + 1)] - ep1) <= radius1)]).sum()
    en1 = sum_amps
    if not silent:
        print("Energy of", ep1, "=", sum_amps)
    sum_amps = np.abs(normed_amps[(np.abs(n1[0: (len(arr) // 2 + 1)] - ep2) <= radius2)]).sum()
    en2 = sum_amps
    if not silent:
        print("Energy of", ep2, "=", sum_amps)
        print("Relation in", ep1, "=", en1 / en2)

    return np.abs(normed_amps), en1, en2, en1 / en2


def draw_feature(data, name, xlim=100, ylim=0):
    """
    Рисует график нужной фичи из таблицы omniweb и результаты разложения в Фурье
            :param ylim: Самая большая амлитуда для отображения в спектре
            :param xlim: Самый большой период для отображения в спектре
            :param name: Название показателя
            :param data: Датафрейм, из которого берутся даные
    """
    if ylim == 0:
        mFFT(arr=data[name], draw=True, name=name, x1=0, x2=xlim)
    else:
        mFFT(arr=data[name], draw=True, name=name, x1=0, x2=xlim, y2=ylim)


def draw_arr_feature(name, data_arr, xlim=100, ylim=0):
    """
    Рисует график по некоторому массиву как по массиву временных точек и его спектр
    Parameters:
        name : string
            Имя для отображения на графике
        data_arr : list
            Массив точек
        xlim : double
            Правая граница периодов для спектра
        ylim : double
            Верхняя граница амплитуд для спектра
    """
    if ylim == 0:
        mFFT(arr=data_arr, draw=True, name=name, x1=0, x2=xlim)
    else:
        mFFT(arr=data_arr, draw=True, name=name, x1=0, x2=xlim, y2=ylim)


def in_delta(x, center, d):
    """
    Проверяет, находится ли точка x на расстоянии не более d от center
    """
    return np.abs(x - center) < d


def in_spring_delta(x, d):
    """ Проверяет, лежит ли точка в окрестности весеннего солнцестояния """
    return in_delta(x, spring, d)


def in_autumn_delta(x, d):
    """ Проверяет, лежит ли точка в окрестности осеннего солнцестояния """
    return in_delta(x, autumn, d)


def imitate_Dst_new(generation, num):
    res = []
    for i in range(num):
        res.append(generation.next())
    return res


def create_generator_type_A(delta=45, T=27, p=1, A_sin=1, D_1=1, B_2=0, D_2=1):
    def foo(day, prev, gen):
        val = p * prev
        freq = 2 * np.pi / T
        if in_autumn_delta(day % DAYS_PER_YEAR, delta) or in_spring_delta(day % DAYS_PER_YEAR, delta):
            val += gen.normal(A_sin * np.sin(day * freq), D_1)
        else:
            val += gen.normal(B_2, D_2)

    return foo


def create_generator_type_B(delta=45, T=27, p=1, A_sin_1=1, D_1=1, A_sin_2=1, D_2=1):
    def foo(day, prev, gen):
        val = p * prev
        freq = 2 * np.pi / T
        if in_autumn_delta(day % DAYS_PER_YEAR, delta) or in_spring_delta(day % DAYS_PER_YEAR, delta):
            val += gen.normal(A_sin_1 * np.sin(day * freq), D_1)
        else:
            val += gen.normal(A_sin_2 * np.sin(day * freq), D_2)

    return foo


def imitate_Dst(Num=19752, delta=45, T=27, p=1, A_sin=1, to_smooth=False, D_1=1, B_2=0, D_2=1, only_sin=False,
                A_sin2=1):
    """
    Продвинутый вариант функции ar(). Разные виды шума добавляются вне delta-окрестности весеннего и осеннего солнцестояний
    и внутри этих окрестностей. Возможно добавлять только синусуоидальный шум, но разный, или же вне окрестности добавлять белый шум
    с параметрами D_2 и B_2, а только внутри синусоидальный.

    Parameters:
        Num : int
             Сколько точек надо сгенерировать
        delta : int
            Размер окрестности равноденствий
        T : int
            Период шума в окрестности равноденствий в днях
        p : double, 0 <= p <= 1
            Параметр p авторегрессионной модели
        A_sin : double
            Амплитуда синуса в шуме окрестности равноденствий
        to_smooth : bool
            Сглаживать ли сгенерированную последовательность (окрестность равноденствий оставляем, как есть, остальные точки заменяем на среднее)
        D_1 : double
            Стандартное отклонение шума в окрестности равноденствий
        B_2 : double
            Среднее шума вне окрестности равноденствий
        D_2 : double
            Стандартное отклонение шума вне окрестности
        only_sin : bool
            Если only_sin == True, то вне окрестности добавляется тоже синусоидальный шум, иначе просто белый с заданными параметрами
        A_sin2 : double
            Работает, если only_sin == True. Амплитуда синусоидального шума вне окрестности равноденствий
        Returns:
            Возвращает массив сгенерированных точек
    """
    freq = 2 * np.pi / T
    def noise_basic(): return generator.normal(B_2, D_2)
    def noise_sin(t): return generator.normal(A_sin * np.sin(freq * t), D_1)
    def noise_sin_small(t): return generator.normal(A_sin2 * np.sin(freq * t), D_2)
    start_value = -20
    arr = [start_value]
    N = Num
    for i in range(N):
        new_val = p * arr[-1]
        day = i % DAYS_PER_YEAR + 1
        if in_spring_delta(day, delta) or in_autumn_delta(day, delta):
            new_val += noise_sin(day)
        else:
            if only_sin:
                new_val += noise_sin_small(day)
            else:
                new_val += noise_basic()
        arr.append(new_val)

    if to_smooth:
        mean_val = np.mean(arr)
        arr1 = []
        for i in range(len(arr)):
            day = i % DAYS_PER_YEAR + 1
            if in_spring_delta(day, delta) or in_autumn_delta(day, delta):
                arr1.append(arr[i])
            else:
                arr1.append(mean_val)
        arr = arr1

    return arr


def smooth_feature(data, name='Dst', delta=45, xlim=360):
    """
    Cглаживает массив data[name] так, чтобы в окрестности равноденствий все оставалось без
    изменений, а вне нее значения заменялось на среднее по всему массиву.
    Parameters:
        data : DataFrame
            Исходный DataFrame
        name : string
            Название колонки, данный в которой надо сгладить
        delta : int
            Окрестность равножженствий
        xlim : int
            Лимит по периодичности в спектре(в днях)
        Returns:
            массив амплитуд спектра + отрисовка графика
    """
    N = data[name].size
    mean_val = np.mean(data[name])
    new_array = []
    for i in range(N):
        if in_spring_delta(data['DOY'].iloc[i], delta) or in_autumn_delta(data['DOY'].iloc[i], delta):
            new_array.append(data[name].iloc[i])
        else:
            new_array.append(mean_val)
    print('mean value =', mean_val)
    return mFFT(np.array(new_array), draw=True, name=name, x2=xlim)


def filt(data, day, rad):
    data['marked'] = (((data['DOY'] - day) % DAYS_PER_YEAR) <= rad) | (
                (data['DOY'] - day) % DAYS_PER_YEAR >= DAYS_PER_YEAR - rad)
    data['windowed'] = data['Dst'] * data['marked'] + (1 - data['marked']) * data['Dst'].mean()


def filt_simple(data, day, rad):
    df = pd.DataFrame(data, columns=['Dst'])
    df['DOY'] = np.arange(0, len(data))
    df['marked'] = (((df['DOY'] - day) % DAYS_PER_YEAR) <= rad) | (
                (df['DOY'] - day) % DAYS_PER_YEAR >= DAYS_PER_YEAR - rad)
    df['windowed'] = df['Dst'] * df['marked'] + (1 - df['marked']) * df['Dst'].mean()
    return df['windowed']


def filt_sim(data, day, rad):
    data['marked'] = (((data['DOY'] - day) % DAYS_PER_YEAR) <= rad) | (
                (data['DOY'] - day) % DAYS_PER_YEAR >= DAYS_PER_YEAR - rad)
    data['fsim'] = data['sim'] * data['marked'] + (1 - data['marked']) * data['sim'].mean()


def retrieve_energy(data, draw=False, steps=73):
    en1s = []
    en2s = []
    rels = []
    for i in range(steps):
        filt(data, 1 + i * (DAYS_PER_YEAR // steps), 30)
        _, en1, en2, rel = mFFTe(data['windowed'], norm=data['marked'].sum())
        en1s.append(en1)
        en2s.append(en2)
        rels.append(rel)
    if draw:
        plt.figure(figsize=(12, 8))
        plt.grid(True)
        plt.plot(np.arange(1, 13.1, 1 / 6), en1s)
        plt.xlabel("month", fontsize=14)
        plt.ylabel("27 days energy", fontsize=14)
        plt.title("Energy change", fontsize=16)
        plt.show()
        
    return max(en1s[:37]), min(en1s[20:50]), max(en1s[37:])


def retrieve_energy_arr(data, draw=False, steps=73):
    en1s = []
    for i in range(steps):
        filt(data, 1 + i * (DAYS_PER_YEAR // steps), 30)
        _, en1, en2, rel = mFFTe(data['windowed'], norm=data['marked'].sum())
        en1s.append(en1)
    if draw:
        plt.plot(np.linspace(1, 13.1, steps), en1s)
        plt.xlabel('month')
        plt.ylabel('27 days energy')
        plt.title('energy change')
        plt.show()
    return en1s


def get_amplitude_of_p(mp, left, right):
    """
    Получить максимальную амплитуду для периодов с left по right при генерации с параметром p=mp
    Все остальные параметры - по умолчанию
    """
    generated_data = imitate_Dst(p=mp, A_sin=1)
    generated_data = np.array(list(map(lambda x: x, generated_data)))
    fft = list(mFFT(generated_data, draw=False, name=''))
    fft_res = list(zip(fft[0], fft[1]))
    filtered = filter(lambda x: left <= x[0] <= right, fft_res)
    return np.max(list(map(lambda x: x[1], filtered)))


def get_semiannual_amp(mp):
    """ Получить полугодовую амплитуду """
    return get_amplitude_of_p(mp, 175, 190)


def get_27day_amp(mp):
    """ Получить 27-дневную амплитуду """
    return get_amplitude_of_p(mp, 24, 30)


def get_relation_of_p(mp, to_smooth=False):
    """ Получить отношение амплитуд """
    generated_data = imitate_Dst(p=mp, to_smooth=to_smooth)
    generated_data = np.array(list(map(lambda x: x, generated_data)))
    tmp = mFFT(arr=generated_data, draw=False, name='')
    fft_res = list(zip(tmp[0], tmp[1]))
    return smoothed_relation(fft_res)


def draw_semiannual_amplitude(p_from=0.5, p_to=1, n=100):
    """ Построить график зависимости полугодовой амплитуды от p """
    plt.figure(figsize=(13, 5))
    plt.xlabel('value of p', fontsize=17)
    plt.ylabel('amplitude of semi-annual period', fontsize=17)
    ps = np.linspace(p_from, p_to, n)
    res = list(map(lambda px: get_semiannual_amp(px), ps))
    plt.title('Dependency of A_semiannual from p', fontsize=19)
    plt.plot(ps, res)


def draw_27day_amplitude(p_from=0.5, p_to=1, n=100):
    """ Построить график зависимости 27-дневной амплитуды от p + линейная аппроксимация"""
    plt.figure(figsize=(13, 5))
    plt.xlabel('value of p', fontsize=17)
    plt.ylabel('amplitude of 27-day period', fontsize=17)
    ps = np.linspace(p_from, p_to, n)
    res = list(map(lambda px: get_27day_amp(px), ps))
    approx_polynom = np.polyfit(ps, res, 1)
    approx = np.polyval(approx_polynom, ps)
    plt.title('Dependency of A_27day from p', fontsize=19)
    plt.plot(ps, res)
    plt.plot(ps, approx)
    print('polynomial : ', approx_polynom)


def draw_amplitude_relation(p_from=0.5, p_to=1, n=100, to_smooth=False):
    plt.figure(figsize=(13, 5))
    plt.xlabel('value of p', fontsize=17)
    plt.ylabel('relation A_27/A_semi ', fontsize=17)
    ps = np.linspace(p_from, p_to, n)
    res = list(map(lambda x: get_relation_of_p(x, to_smooth), ps))
    plt.title('Dependency of amplitude relationship from p', fontsize=19)
    plt.plot(ps, res)


def smooth_amps(t_from, t_to, amps):
    diap = list(filter(lambda x: t_from <= x[0] <= t_to, amps))
    return sum(map(lambda x: x[1], diap))


def smooth_27(amps):
    return smooth_amps(27 - 4, 27 + 4, amps)


def smooth_183(amps):
    return max(map(lambda x: x[1], filter(lambda x: 183 - 4 <= x[0] <= 183 + 4, amps)))


def smoothed_relation(amps):
    return smooth_27(amps) / max(0.1, smooth_183(amps))


def get_energys(data, doys, draw=False):
    en1s = []
    en2s = []
    rels = []
    step = 5
    iterations = DAYS_PER_YEAR // step
    for i in range(iterations):
        day = 1 + i * step
        rad = 30
        marked = (((doys - day) % DAYS_PER_YEAR) <= rad) | ((doys - day) % DAYS_PER_YEAR >= DAYS_PER_YEAR - rad)
        windowed = data * marked + (1 - marked) * data.mean()
        _, en1, en2, rel = mFFTe(windowed, norm=marked.sum())
        en1s.append(en1)
        en2s.append(en2)
        rels.append(rel)
    if draw:
        plt.plot(np.linspace(1, 13.1, iterations), en1s)
        plt.xlabel('month')
        plt.ylabel('27 days energy')
        plt.title('energy change')
        plt.show()
    return en1s


def draw_two(data1, doys1, data2, doys2):
    en1s = get_energys(data1, doys1)
    en2s = get_energys(data2, doys2)
    step = 5
    plt.plot(np.linspace(1, 13.1, DAYS_PER_YEAR // step), en1s)
    plt.plot(np.linspace(1, 13.1, DAYS_PER_YEAR // step), en2s)
    plt.xlabel('month')
    plt.ylabel('27 days energy')
    plt.title('energy change')
    plt.show()


class collector:
    def __init__(self, n):
        self.arr = [[] for i in range(n)]
        self.counter = 0
        self.averaged = []
        self.mins = []

    def add(self, arrx):
        i = 0
        for t in arrx:
            self.arr[i].append(t)
            i += 1
        self.counter += 1

    def getAveraged(self):
        if self.counter == 0:
            return np.array(1)
        self.averaged = []
        for a in self.arr:
            np_array = np.array(a)
            self.averaged.append((np.average(np_array), np.std(np_array)))
        return self.averaged

    def getMin(self):
        if self.counter == 0:
            return np.array(1)
        self.mins = []
        for a in self.arr:
            np_array = np.array(a)
            if len(np_array) == 0:
                self.mins.append((0, 0, 0))
            else:
                self.mins.append((np.min(np_array), np.average(np_array), np.max(np_array)))
        return self.mins


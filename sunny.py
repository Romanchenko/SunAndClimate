# imports
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import math

# set random generator
generator = np.random.RandomState(42)

# settings of graphics
width = 17
height = 5
matplotlib.rcParams['figure.figsize'] = [width, height]
matplotlib.rcParams['lines.markersize'] = 0.5
#matplotlib.rcParams['scatter.edgecolors'] = "black"

#  days of spring and autumn equinoxes
spring = 79
autumn = 266

# autoregression generator
def ar(p, base_value, noise_maker, to_add = 10000):
    """
    Генератор последовательности по авторегрессионной модели
    Parameters
    ----------
        p : double
            Параметр памяти авторегрессионной модели. 0 <= p <= 1
        base_value : double
            Первое значение, с которого начинается генерация
        noise_masker : function
            Функция, возвращающая случайное значение - шум
        to_add : int
            Сколько элементов должно быть сгенерировано
    Returns
    -------
        Массив(list) сгенерированных точек
    """

    new_vals = np.array(arr)
    new_vals = np.append(new_vals, [0] * to_add)
    pos = 1
    for i in range(to_add):
        noise = noise_maker(i)
        new_vals[pos + i] = p * new_vals[pos + i - 1] + noise
    return new_vals

def mFFT(arr, draw=False, name='a[n]', x1 = 0, x2 = -1, y2 = -1, energy=False, norm=-1, smth=0, silent=True):
    """
    Parameters
    ----------
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
    smth : bool, optional
        Наличие сглаживания
    norm : double
        Число для нормировки, стандартное - длина массива
    energy : bool
        Искать ли энергию точки
    ep : double
        Точка для поиска энергии
    radius : double
        Радиус энергии
    silent : bool
        Нужен ли отладочный вывод
        
    Returns
    -------
    Если не выставлен флаг enerhy, то возвращается массив абсолютных значений амплитуд спектра.
    Если флаг выставлен, то возращается кортеж из четырех элементов - энергия амплитуды 27-дневного сигнала,
    энергия амплитуды полугодового сигнала, их отношение и массив абсолютных значений амплитуд сигнала.
    """
    if x2 == -1:
        x2 = len(arr) // 2
    if norm == -1:
        norm = len(arr)
        
    
    A = np.fft.rfft((arr - np.mean(arr)) / norm)
    
    if draw:
        
        plt.rcParams['axes.grid'] = True
        fig, ax = plt.subplots(2, figsize=(6,4), dpi=150)
        plt.tight_layout()
        
        n = np.arange(len(arr))
        n1 = len(arr) / n[1:]
        
        plt.subplots_adjust(hspace=0.5)
        
        ax[0].plot(n, arr, '.-')
        ax[0].set_title(name) 
        
        ax[1].set_title('$A$')
        if y2 > 0:
            ax[1].set_ylim(0, y2)
        ax[1].set_xlim(x1, x2)
        if smth > 0:
            ax[1].plot(n1[0 : (len(arr) // 2 - 2 * smth + 1)], smoth(np.abs(A), smth), '-')
        else:
            ax[1].plot(n1[0 : (len(arr) // 2 + 1)], np.abs(A), '-')
        ax[1].set_xlabel('Период (в днях)')
        
        plt.show()
    
    if (energy):
        n = np.arange(len(arr))
        n1 = len(arr) / n[1:]
        
        ep = 29
        radius = 3
        
        msum = np.abs(A[(np.abs(n1[0 : (len(arr) // 2 + 1)] - ep) <= radius)]).sum()
        en1 = msum
        if not silent:
            print("Energy of", ep, "=", msum )
        
        ep = 183
        radius = 2
        
        msum = np.abs(A[(np.abs(n1[0 : (len(arr) // 2 + 1)] - ep) <= radius)]).sum()
        en2 = msum
        if not silent:
            print("Energy of", ep, "=", msum)
            print("Relation in", ep, "=", en1/en2)
        
        return en1, en2, en1/en2, np.abs(A)
    
    return (np.abs(A), n1)


def draw_feature(name, xlim=100, ylim=0):
    """
    Рисует график нужной фичи из таблицы omniweb и результаты разложения в Фурье
    Parameters:
    ----------
        name : srting
            Название показателя
        xlim : double
            Самый большой период для отображения в спектре
        ylim : double
            Самая большая амлитуда для отображения в спектре
    """
    if ylim == 0:
        mFFT(arr=data[name], draw=True, name=name, x1=0, x2=xlim)
    else:
        mFFT(arr=data[name], draw=True, name=name, x1=0, x2=xlim, y=ylim)

def draw_arr_feature(name, data_arr, xlim=100, ylim=0):
    """
    Рисует график по некоторому массиву как по массиву временных точек и его спектр
    Parametrs:
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
        mFFT(arr=data_arr, draw=True, name=name, x1=0, x2=xlim, y=ylim)



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

def imitate_Dst(delta=45, T=27, p=1, A_sin=1, to_smooth=False, D_1=1, B_2=0, D_2=1, only_sin=False, A_sin2=1):
    """
    Продвинутый вариант @link(ar). Разные виды шума добавляются вне delta-окрестности весеннего и осеннего солнцестояний
    и внутри этих окрестностей. Возможно добавлять только синусуоидальный шум, но разный, или же вне окрестности добавлять белый шум
    с параметрами D_2 и B_2, а только внутри синусоидальный.
    Parameters:
    ----------
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
        -------
            Возвращает массив сгенерированных точек
    """
    freq = 2 * np.pi / T
    noiser_basic = lambda : generator.normal(B_2, D_2)
    noiser_sin = lambda i : generator.normal(A_sin * np.sin(freq * i), D_1)
    noiser_sin_small = lambda i : generator.normal(A_sin2 * np.sin(freq * i), D_2)
    start_value = -20
    arr = [start_value]

    for i in range(N):
        new_val = p * arr[-1]
        day = i % 365 + 1
        if in_spring_delta(day, delta) or in_autumn_delta(day, delta):
            new_val += noiser_sin(day)
        else:
            if only_sin:
                new_val += noiser_sin_small(day)
            else:
                new_val += noiser_basic()
        arr.append(new_val)
    
    if to_smooth:
        mean_val = np.mean(arr)
        arr1 = []
        for i in range(len(arr)):
            day = i % 365 + 1
            if in_spring_delta(day, delta) or in_autumn_delta(day, delta):
                arr1.append(arr[i])
            else:
                arr1.append(mean_val)
        arr = arr1

    return arr

def smooth_feature(data, name="Dst", delta=45, xlim=360):
    """
    Cглаживает массив data[name] так, чтобы в окрестности равноденствий все оставалось без
    изменений, а вне нее значения заменялось на среднее по всему массиву.
    Parameters:
    ----------
        data : DataFrame
            Исходный DataFrame
        name : string
            Название колонки, данный в которой надо сгладить
        delta : int
            Окрестность равножженствий
        xlim : int
            Лимит по периодичности в спектре(в днях)
        Returns:
        -------
            массив амплитуд спектра + отрисовка графика
    """
    N = data[name].size
    arr = np.array(data[name])
    mean_val = np.mean(data[name])
    narr = []
    for i in range(N):
        if in_spring_delta(data['Day'].iloc[i], delta) or in_autumn_delta(data['Day'].iloc[i], delta):
            narr.append(data[name].iloc[i])
        else:
            narr.append(mean_val)
    print('mean value =', mean_val)
    return mFFT(narr, draw=True, name=name, x2=xlim)

def filt(data, day, rad):
    data["filt"] = (((data["Day"] - day) % 365) <= rad) | ((data["Day"] - day) % 365 >= 365 - rad)
    data["filted"] = data["Dst"] * data["filt"] + (1 - data["filt"]) * data["Dst"].mean()

def filt_simple(data, day, rad):
    df = pd.DataFrame(data, columns=["Dst"])
    df['Day'] = np.arange(0, len(data))
    df["filt"] = (((df["Day"] - day) % 365) <= rad) | ((df["Day"] - day) % 365 >= 365 - rad)
    df["filted"] = df["Dst"] * df["filt"] + (1 - df["filt"]) * df["Dst"].mean()
    return df["filted"]
    
def get_year_max(datax, day, rad, year):
    datax["filt"] = (
        (abs(365 + datax["Day"] - day) % 365 <= rad)
        & (abs(datax['Year'] - year - 1965) < 1))
    return max(datax["Dst"] * datax["filt"])

def retrieve_energy(ddata, draw=False):
    en1s = []
    en2s = []
    rels = []
    for i in range(73):
        filt(ddata, 1 + i * 5, 30)
        en1, en2, rel, _ = mFFT(ddata["filted"], draw=False, x1=20, x2=40, norm=ddata["filt"].sum(), energy=True)
        en1s.append(en1)
        en2s.append(en2)
        rels.append(rel)
    if draw:
        plt.plot( np.arange(1, 13.1, 1 / 6), en1s)
        plt.xlabel("month")
        plt.ylabel("27 days energy")
        plt.title("energy change")
        plt.show()
    return (max(en1s[:37]), min(en1s[20:50]), max(en1s[37:]))
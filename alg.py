import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import math
from random import random
import pandas as pd
import matplotlib.pyplot as plt
import functools
from scipy.optimize import minimize_scalar


def func_1():
    """Функция из первой ячейки, подбирает АРМА-параметры к синусоиде со случайным шумом"""
    n_max = 102
    # Задаем ряд
    price_ser_src = pd.Series([sin(x) + 10 * random() for x in range(1, n_max)])
    # p_ser = price_ser_src[::5]
    # Находим mu (тренд)
    # Простой способ -- первая разность
    mu_arr = np.array([price_ser_src[i] - price_ser_src[i - 1] if i > 0 else 0. for i in range(len(price_ser_src))])
    # Находим параметры ARMA (подгоняем модель)
    model = ARIMA(price_ser_src, order=(4, 0, 4))
    model_fit = model.fit()
    print(model_fit.summary())
    # make prediction -- это пока вообще не трогаем
    # yhat = model_fit.predict(len(data), len(data))
    # print(yhat)
    # График
    plt.title("Временной ряд")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("p_n")  # ось ординат
    plt.grid()  # включение отображение сетки
    # plt.plot(p_ser)  # построение графика
    plt.plot(price_ser_src)
    plt.plot(mu_arr)
    plt.show()


def create_arma44_ser():
    """Генерирует ряд ARMA(4,4), необходимый для тестирования алгоритмов.
    В версии 0.1 возвращает кортеж из Numpy-массивов p (цена) и mu (тренд) и рисует их в matplotlib.
    mu считается по первой разности цены, mu[i] = p[i]-p[i+1], mu[0]=0"""
    n_max = 120
    # Параметры ARMA-ряда, экспертные от Вадима А.
    a = [10., 8., 6., 2., .1]
    d = [1., .6, .4, .1, .06]
    p = [-1., 3., .1, -.8, .5]
    for i in range(5, n_max + 5):
        p_next = a[0] * p[4] + a[1] * p[3] + a[2] * p[2] + a[3] * p[1] + a[4] * p[0]
        xi = np.random.normal(0., 10., 5)
        p_next += sum([d[i] * xi[i] for i in range(5)])
        p.append(p_next)
    p_arr_arma = np.array(p[5:])
    mu_arr_arma = np.diff(p_arr_arma)
    # Сдвигаем разницу на одну единицу вперед, чтобы выполнялось условие mu[i] = p[i] - p[i+1]
    mu_arr_arma = np.append(0., mu_arr_arma)
    n_lst = list(range(n_max))
    figure = plt.figure(figsize=(8, 6))
    plt.title("ARMA (4, 4) series")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("p_n")  # ось ординат
    plt.grid()  # включение отображение сетки
    # plt.plot(p_ser)  # построение графика
    plt.plot(n_lst, p_arr_arma, mu_arr_arma)
    plt.legend(['p_n', 'mu'], loc="upper left")
    plt.show()
    return p_arr_arma, mu_arr_arma


def load_test_data():
    """
    Загружает из csv, лежащих в папке проекта, данные (биржевые котировки за 4 дня в феврале 2022 года)
    для тестирования алгоритмов. Возвращает кортеж из массива и тренда за каждую дату, всего 8 np.array-ев.
    """
    col_names = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7']

    df_stable = pd.read_csv('trades-16-02-22.csv', sep=';', names=col_names)
    col3_ser_stable = df_stable['col3']
    col3_ser_stable.reset_index(drop=True, inplace=True)

    df_growth = pd.read_csv('trades-17-02-22.csv', sep=';', names=col_names)
    col3_ser_growth = df_growth['col3']
    col3_ser_growth.reset_index(drop=True, inplace=True)

    df_si_14 = pd.read_csv('trades-si-14-02-22.csv', sep=';', names=col_names)
    col3_ser_si_14 = df_si_14['col3']
    col3_ser_si_14.reset_index(drop=True, inplace=True)

    df_si_21 = pd.read_csv('trades-si-21-02-22.csv', sep=';', names=col_names)
    col3_ser_si_21 = df_si_21['col3']
    col3_ser_si_21.reset_index(drop=True, inplace=True)

    price_series_stable = np.array(col3_ser_stable)
    p_arr_stable = np.array([price_series_stable[i] for i in range(len(price_series_stable)) if i % 5000 == 0])
    p_init_stable = price_series_stable[0]
    mu_arr_stable = np.diff(p_arr_stable)
    # Сдвигаем разницу на одну единицу вперед (чтобы тренд был в конце интервала, а не в начале)
    mu_arr_stable = np.append(0., mu_arr_stable)

    price_series_growth = np.array(col3_ser_growth)
    p_arr_growth = np.array([price_series_growth[i] for i in range(len(price_series_growth)) if i % 5000 == 0])
    p_init_growth = price_series_growth[0]
    mu_arr_growth = np.diff(p_arr_growth)
    mu_arr_growth = np.append(0., mu_arr_growth)

    price_series_si_14 = np.array(col3_ser_si_14)
    p_arr_si_14 = np.array([price_series_si_14[i] for i in range(len(price_series_si_14)) if i % 5000 == 0])
    p_init_si_14 = price_series_si_14[0]
    mu_arr_si_14 = np.diff(p_arr_si_14)
    mu_arr_si_14 = np.append(0., mu_arr_si_14)

    price_series_si_21 = np.array(col3_ser_si_21)
    p_arr_si_21 = np.array([price_series_si_21[i] for i in range(len(price_series_si_21)) if i % 5000 == 0])
    p_init_si_21 = price_series_si_21[0]
    mu_arr_si_21 = np.diff(p_arr_si_21)
    mu_arr_si_21 = np.append(0., mu_arr_si_21)

    """
    n_lst = list(range(len(p_arr_stable)))
    plt.subplot(2, 5, 2)
    plt.title("Real data, price USD-RUB 16.02.2022")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("p_n")  # ось ординат
    plt.grid()  # включение отображение сетки
    # plt.plot(p_ser)  # построение графика
    plt.plot(n_lst, p_arr_stable)
    plt.legend(['p_n'], loc="upper left")

    plt.subplot(2, 5, 7)
    plt.title("Trend modeling by first difference")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("mu")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(n_lst, mu_arr_stable, color='orange')
    plt.legend(['mu'], loc="upper left")

    n_lst = list(range(len(p_arr_growth)))
    plt.subplot(2, 5, 3)
    plt.title("Real data, price USD-RUB 17.02.2022")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("p_n")  # ось ординат
    plt.grid()  # включение отображение сетки
    # plt.plot(p_ser)  # построение графика
    plt.plot(n_lst, p_arr_growth)
    plt.legend(['p_n'], loc="upper left")

    plt.subplot(2, 5, 8)
    plt.title("Trend modeling by first difference")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("mu")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(n_lst, mu_arr_growth, color='orange')
    plt.legend(['mu'], loc="upper left")

    n_lst = list(range(len(p_arr_si_14)))
    plt.subplot(2, 5, 4)
    plt.title("Real data, price USD-RUB 14.02.2022")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("p_n")  # ось ординат
    plt.grid()  # включение отображение сетки
    # plt.plot(p_ser)  # построение графика
    plt.plot(n_lst, p_arr_si_14)
    plt.legend(['p_n'], loc="upper left")

    plt.subplot(2, 5, 9)
    plt.title("Trend modeling by first difference")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("mu")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(n_lst, mu_arr_si_14, color='orange')
    plt.legend(['mu'], loc="upper left")

    n_lst = list(range(len(p_arr_si_21)))
    plt.subplot(255)
    plt.title("Real data, price USD-RUB 21.02.2022")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("p_n")  # ось ординат
    plt.grid()  # включение отображение сетки
    # plt.plot(p_ser)  # построение графика
    plt.plot(n_lst, p_arr_si_21)
    plt.legend(['p_n'], loc="upper left")

    plt.subplot(2, 5, 10)
    plt.title("Trend modeling by first difference")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("mu")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(n_lst, mu_arr_si_21, color='orange')
    plt.legend(['mu'], loc="upper left")
    # fig.subplots_adjust(hspace=.2)
    plt.show()"""
    return p_arr_stable, mu_arr_stable, \
           p_arr_growth, mu_arr_growth, \
           p_arr_si_14, mu_arr_si_14, \
           p_arr_si_21, mu_arr_si_21


def minus_eg(k, _d_inv, _mu, _delta_t):
    """Функция Eg, которую мы максимизируем, взятая со знаком минус, потому что scipy.optimize ищет минимум."""
    res = -_mu / math.fabs(_mu) * _d_inv / k * (math.exp(k * _mu * _delta_t) + math.exp(-k * _mu * _delta_t) - 2.)
    return res


def calc_alg0(_p_arr, _mu_arr, di0):
    """Расчет по алгоритму alg 0.5, на входе массивы p, mu и начальная инвестиция
    Учитываем лонги/шорты
    TODO: коммичу в отдельный бранч в гите"""

    di_norm = 1000000.

    n_max = len(_p_arr) - 1
    # Ограничения на K
    k_min = -10.
    k_max = 10.
    # Массивы
    t_max = float(n_max)  # /100
    t_arr = np.linspace(0., t_max, n_max + 1)
    dt = 1.
    k_arr = np.zeros(n_max + 1)
    dg_arr = np.zeros(n_max + 1)
    di_arr = np.zeros(n_max + 1)
    di_arr[1] = di0 / di_norm
    # print('Starting alg0, I_0 = ', di0)

    history = []

    long_is_opened = False
    short_is_opened = False
    price_prev = _p_arr[0]

    for i in range(1, n_max-1):



        if i == 24:
            q = 1
            pass




        des_str = ''
        mu_factor = math.fabs(_mu_arr[i])/max(np.abs(_mu_arr[:i+1]))
        mu_normalized = _mu_arr[i] / 10.
        if short_is_opened:
            dg_pos = - (_p_arr[i] - price_prev)
            if dg_pos > 5. or len(history) > 1 or i == len(_p_arr)-1:
                des_str = 'Close short'
                short_is_opened = False
                k_cur = 10.
                k_arr[i] = k_cur
                dg_arr[i] = dg_pos
                di_arr[i+1] = k_arr[i] * dg_arr[i] / di_norm
                price_prev = _p_arr[i]
            else:
                k_arr[i] = 0.
                des_str = 'Hold'
                dg_arr[i] = 0.
                di_arr[i + 1] = di_arr[i]
                di_arr[i] = 0.
                history.append(dg_pos)
        elif long_is_opened:
            dg_pos = _p_arr[i] - price_prev
            if dg_pos > 5. or len(history) > 1 or i == len(_p_arr)-3:
                des_str = 'Close long'
                long_is_opened = False
                k_cur = -10.
                k_arr[i] = k_cur
                dg_arr[i] = dg_pos
                di_arr[i+1] = k_arr[i] * dg_arr[i] / di_norm
                price_prev = _p_arr[i]
            else:
                k_arr[i] = 0.
                des_str = 'Hold'
                dg_arr[i] = 0.
                di_arr[i + 1] = di_arr[i]
                di_arr[i] = 0.
                history.append(dg_pos)
        elif math.fabs(_mu_arr[i]) < .1:
            # di_arr[i] == ? (это определено на прошлом шаге)
            k_arr[i] = 0.
            des_str = 'Hold'
            # А можно и закрыть позицию!
            dg_arr[i] = 0.
            di_arr[i+1] = di_arr[i]
            di_arr[i] = 0.
            price_prev = _p_arr[i]
        elif math.fabs(_mu_arr[i]) > 30.:
            k_arr[i] = 0.
            des_str = 'Hold'
            dg_arr[i] = 0.
            di_arr[i+1] = di_arr[i]
            di_arr[i] = 0.
            price_prev = _p_arr[i]
        # elif i > 2 and dg_arr[i-1] < dg_arr[i-3] < 0.:
        #    k_arr[i] = 0.
        #    des_str = 'Hold'
        #    dg_arr[i] = 0.
        #    di_arr[i + 1] = di_arr[1]
        #    di_arr[i] = 0.
        else:
            # di_arr[i] == ? (это определено на прошлом шаге)
            minus_eg_cur = functools.partial(minus_eg, _d_inv=di_arr[i], _mu=mu_normalized, _delta_t=dt)
            k_cur = - minimize_scalar(minus_eg_cur, bounds=(k_min, k_max), method='bounded').x
            k_arr[i] = k_cur

            # if dg_arr[i] < 20:
            #    k_arr[i] = 0.
            #    dg_arr[i+1] = 0.
            #    di_arr[i+1] = 0.
            #    des_str = 'Nothing'
            # else:

            if k_cur < 0:
                des_str = 'Open short'
                short_is_opened = True
                history = []
                price_prev = _p_arr[i]
                dg_arr[i] = 0
                di_arr[i+1] = -di_arr[i]
            else:
                des_str = 'Open long'
                history = []
                long_is_opened = True
                price_prev = _p_arr[i]
                dg_arr[i] = 0
                di_arr[i+1] = -di_arr[i]
        if True:
            print(f'iter={i}, t={round(t_arr[i], 2)}, p={_p_arr[i]}, mu_n={round(mu_normalized, 4)}',
                  f'mu_factor={round(mu_factor, 2)}, dI={di_arr[i]}, dg={dg_arr[i]}, '
                  f'K={round(k_arr[i], 4)} -> {des_str}')
            pass

    dk_ser = pd.Series(k_arr[:-2], index=t_arr[:-2])
    dg_ser = pd.Series(dg_arr[:-2], index=t_arr[:-2])
    di_ser = pd.Series(di_arr[:-2]*di_norm, index=t_arr[:-2])
    profit_ser = np.cumsum(dg_ser[:n_max - 2])
    profit = profit_ser[len(profit_ser)-1]
    print('Заработано:', profit)
    plt.figure(figsize=(20, 3))
    n_lst = list(range(len(_p_arr)))
    plt.subplot(141)
    plt.title("Price")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("p_n")  # ось ординат
    plt.grid()  # включение отображение сетки
    # plt.plot(p_ser)  # построение графика
    plt.plot(n_lst, _p_arr)
    plt.legend(['p_n'], loc="upper left")
    plt.subplot(142)
    plt.plot(dg_ser[:n_max - 2], marker='o')
    plt.xlabel("t")
    plt.ylabel("dg")
    plt.title("Current profit")
    plt.grid()
    plt.subplot(143)
    plt.plot(di_ser[:n_max - 2], marker='o')
    plt.xlabel("t")
    plt.ylabel("dI")
    plt.title("Current investment")
    plt.grid()
    plt.subplot(144)
    plt.plot(di_ser[:n_max - 2], marker='o')
    plt.xlabel("t")
    plt.ylabel("dI_log")
    plt.title("Logarythmic current investment")
    plt.yscale('log')
    plt.grid()
    plt.show()

    plt.figure(figsize=(20, 3))
    # plt.tight_layout()
    plt.subplot(141)
    plt.title("Trend modeling by first difference")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("mu")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(n_lst, _mu_arr, color='orange')
    plt.legend(['mu'], loc="upper left")
    plt.subplot(142)
    plt.plot(np.cumsum(dg_ser[:n_max - 2]), color='orange')
    plt.xlabel("t")
    plt.ylabel("g")
    plt.title("Cumulative profit")
    plt.grid()
    plt.subplot(143)
    plt.plot(np.cumsum(di_ser[:n_max - 2]), color='orange')
    plt.xlabel("t")
    plt.ylabel("I")
    plt.title("Cumulative investment")
    plt.grid()
    plt.subplot(144)
    plt.plot(k_arr[:n_max - 2])
    plt.xlabel("t")
    plt.ylabel("K")
    plt.title("(technical plot 1)")
    plt.grid()
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.show() 

    figure = plt.figure(figsize=(10, 3))
    plt.title("Decision Buy/Sell/Do nothing (zoomed)")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("K_n")  # ось ординат
    plt.grid()  # включение отображение сетки
    # plt.plot(p_ser)  # построение графика
    plt.plot(n_lst[50:101], k_arr[50:101])
    plt.plot(n_lst[50:101], (_p_arr[50:101] - _p_arr.mean()) / _p_arr.std() * 50)
    plt.legend(['k_arr', 'p_n'], loc="upper left")
    return profit



def calc_alg2(_p_arr, _mu_arr, di0):
    """Расчет по алгоритму alg 2.1, на входе массивы p, mu и начальная инвестиция"""
    n_max = len(_p_arr) - 1
    # Ограничения на K
    k_min = -10
    k_max = 10
    # Массивы
    t_max = float(n_max)  # /100
    t_arr = np.linspace(0., t_max, n_max + 1)
    # dt = t_arr[1]-t_arr[0]
    dt = 1
    k_arr = np.zeros(n_max + 1)
    dg_arr = np.zeros(n_max + 1)
    di_arr = np.zeros(n_max + 1)
    di_arr[0] = di0
    # print('Starting alg0, I_0 = ', di0)
    for i in range(1, n_max - 1):
        des_str = ''


        # if i==20:
        #    print('_mu_arr[20] =', _mu_arr[20])


        if math.fabs(_mu_arr[i]) <= .1:
            k_arr[i] = 0.
            dg_arr[i] = 0.
            des_str = 'Nothing'
            di_arr[i + 1] = 0.
        else:
            mu_normalized = _mu_arr[i] / 10
            minus_eg_cur = functools.partial(minus_eg, _d_inv=di_arr[i], _mu=mu_normalized, _delta_t=dt)
            k_cur = minimize_scalar(minus_eg_cur, bounds=(k_min, k_max), method='bounded').x
            # Корректируем k по значению mu: умножаем его на |mu/12| (mu до нормализации)
            k_cur *= math.fabs(mu_normalized) / 30
            k_arr[i] = k_cur
            # dg_arr[i] = k_cur / math.fabs(k_cur) * (_p_arr[i] - _p_arr[i - 1])
            dg_arr[i] = k_cur * (_p_arr[i] - _p_arr[i - 1])

            if dg_arr[i] < 20.:

                # if i == 20:
                #    print('I\'m here!')

                k_arr[i] = 0.
                dg_arr[i] = 0.
                di_arr[i+1] = 0.
                des_str = 'Nothing'
            else:
                if k_cur >= 0:
                    des_str = 'Buy'
                else:
                    des_str = 'Sell'

            # di_arr[i + 1] = di_arr[i] + k_arr[i] * dg_arr[i]

            def sign(x):
                if math.fabs(x) > 1.e-6:
                    return x / math.fabs(x)
                else:
                    return 0.

            di_arr[i + 1] = k_arr[i] * dg_arr[i]


            # if i == 19:
            #    print('di[19] =', di_arr[i+1])



            if i <= 30:
                # print(f'iter={i}, t={round(t_arr[i], 2)}, dI={round(di_arr[i], 4)}, mu={round(mu_normalized, 4)}',
                #      f'g={round(dg_arr[i], 4)}, K={round(k_arr[i], 4)} -> {des_str}')
                pass
    dk_ser = pd.Series(k_arr[:-2], index=t_arr[:-2])
    dg_ser = pd.Series(dg_arr[:-2], index=t_arr[:-2])
    di_ser = pd.Series(di_arr[:-2], index=t_arr[:-2])
    # print(di_arr)
    plt.figure(figsize=(20, 3))

    n_lst = list(range(len(_p_arr)))
    plt.subplot(141)
    plt.title("Price")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("p_n")  # ось ординат
    plt.grid()  # включение отображение сетки
    # plt.plot(p_ser)  # построение графика
    plt.plot(n_lst, _p_arr)
    plt.legend(['p_n'], loc="upper left")
    plt.subplot(142)
    plt.plot(dg_ser[:n_max - 2], marker='o')
    plt.xlabel("t")
    plt.ylabel("dg")
    plt.title("Current profit")
    plt.grid()
    plt.subplot(143)
    plt.plot(di_ser[:n_max - 2], marker='o')
    plt.xlabel("t")
    plt.ylabel("dI")
    plt.title("Current investment")
    plt.grid()
    plt.subplot(144)
    plt.plot(k_arr[:n_max - 2])
    plt.xlabel("t")
    plt.ylabel("K")
    plt.title("PID Gain coefficient")
    plt.grid()
    plt.show()

    plt.figure(figsize=(20, 3))
    # plt.tight_layout()
    plt.subplot(141)
    plt.title("Trend modeling by first difference")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("mu")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(n_lst, _mu_arr, color='orange')
    plt.legend(['mu'], loc="upper left")
    plt.subplot(142)
    plt.plot(np.cumsum(dg_ser[:n_max - 2]), color='orange')
    plt.xlabel("t")
    plt.ylabel("g")
    plt.title("Cumulative profit")
    plt.grid()
    plt.subplot(143)
    plt.plot(np.cumsum(di_ser[:n_max - 2]), color='orange')
    plt.xlabel("t")
    plt.ylabel("I")
    plt.title("Cumulative investment")
    plt.grid()
    plt.subplot(144)
    plt.plot(k_arr[:n_max - 2])
    plt.xlabel("t")
    plt.ylabel("K")
    plt.title("(technical plot 1)")
    plt.grid()
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    figure = plt.figure(figsize=(10, 3))
    plt.title("Decision Buy/Sell/Do nothing (zoomed)")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("K_n")  # ось ординат
    plt.grid()  # включение отображение сетки
    # plt.plot(p_ser)  # построение графика
    plt.plot(n_lst[50:101], k_arr[50:101])
    plt.plot(n_lst[50:101], (_p_arr[50:101] - _p_arr.mean()) / _p_arr.std() * 50)
    plt.legend(['k_arr', 'p_n'], loc="upper left")


def calc_alg5(_p_arr: np.array, output_flag=True, trend=False, i_const=0., d_const=0., dd_const=0.):
    """Полный PIDD-регулятор"""
    """Расчет по алгоритму alg 0.5, на входе массивы p, mu и начальная инвестиция
        Учитываем лонги/шорты
        TODO: коммичу в отдельный бранч в гите"""
    # Время для биржи с фьючерсом рубль-доллар -- 14 часов
    # Для биткоина будет 24 часа
    n_max = len(_p_arr)
    t_max = 14. * 3600
    if n_max > 500:
        t_max = 24. * 3600
    t_arr = np.linspace(0., t_max, n_max)
    dt = t_arr[1] - t_arr[0]
    # Настроечные параметры
    mu_min = .1
    mu_max = 30.
    if trend:
        mu_min /= dt
        mu_max /= dt
    # Для интегрального члена
    dg_diff_prev = 0.
    mu_arr = np.zeros(n_max)
    if not trend:
        mu_arr = np.array([_p_arr[i]-_p_arr[i-1] if i > 0 else 0. for i in range(n_max)])
    dmu_arr = np.zeros(n_max)
    k_arr = np.zeros(n_max)
    k_i_arr = np.zeros(n_max)
    k_d_arr = np.zeros(n_max)
    k_dd_arr = np.zeros(n_max)
    dg_arr = np.zeros(n_max)
    di_arr = np.zeros(n_max)
    di_arr[1] = 1.
    history = []
    long_is_opened = False
    short_is_opened = False
    price_prev = _p_arr[0]
    # Main cycle
    for i in range(1, n_max-1):
        if trend and i > 5:
            _, gdp_trend = sm.tsa.filters.hpfilter(_p_arr[:i+1])
            mu_arr[i] = (gdp_trend[i] - gdp_trend[i-1]) / dt
            dmu_arr[i] = (mu_arr[i] - mu_arr[i-1]) / dt
        des_str = ''
        k = 0.
        if i == n_max-1:
            k_arr[i], k_i_arr[i], k_d_arr[i], k_dd_arr[i], dg_arr[i], di_arr[i] = 0., 0., 0., 0., 0., 0.
            break
        # Integral part
        integ_sum = 0.
        k_i_arr[i] = i_const
        if i > 10:
            coef_arr = np.array([math.exp(float(-j)/10.) for j in range(10, -1, -1)])
            for cnt in range(10, -1, -1):
                # print(i, cnt, i-cnt, 10-cnt)
                integ_sum += dg_arr[i - cnt] * coef_arr[10 - cnt]
        # Differential part
        k_d_arr[i] = d_const
        # Double-differential part
        k_dd_arr[i] = dd_const
        # Proportional part
        k_p_const = 1.
        # Decision part
        if short_is_opened:
            dg_pos = -(_p_arr[i] - price_prev)
            if dg_pos > 0. or len(history) > 1:
                des_str = 'Close short'
                short_is_opened = False
                k_arr[i] = 1.
                dg_arr[i] = dg_pos
                dg_diff = (dg_arr[i] - dg_arr[i-1]) / dt
                dg_diff_diff = (dg_diff - dg_diff_prev) / dt
                dg_diff_prev = dg_diff
                di_arr[i + 1] = \
                    k_arr[i] * dg_arr[i] + \
                    k_i_arr[i] * integ_sum + \
                    k_d_arr[i] * dg_diff + \
                    k_dd_arr[i] * dg_diff_diff
                price_prev = _p_arr[i]
            else:
                k_arr[i] = 0.
                des_str = 'Hold'
                dg_arr[i] = 0.
                di_arr[i+1] = 1.
                history.append(dg_pos)
        elif long_is_opened:
            dg_pos = _p_arr[i] - price_prev
            if dg_pos > 0. or len(history) > 1:
                des_str = 'Close long'
                long_is_opened = False
                k_arr[i] = -1.
                dg_arr[i] = dg_pos
                dg_diff = (dg_arr[i] - dg_arr[i-1]) / dt
                dg_diff_diff = (dg_diff - dg_diff_prev) / dt
                dg_diff_prev = dg_diff
                di_arr[i + 1] = \
                    k_arr[i] * dg_arr[i] + \
                    k_i_arr[i] * integ_sum + \
                    k_d_arr[i] * dg_diff + \
                    k_dd_arr[i] * dg_diff_diff
                price_prev = _p_arr[i]
            else:
                des_str = 'Hold'
                k_arr[i] = 0.
                dg_arr[i] = 0.
                di_arr[i+1] = 1.
                history.append(dg_pos)
        elif math.fabs(mu_arr[i]) > mu_max:
            des_str = 'Hold'
            k_arr[i] = 0.
            dg_arr[i] = 0.
            di_arr[i+1] = 1.
        else:
            if di_arr[i] > 0. * mu_arr[i] > 0.:
                des_str = 'Open long'
                history = []
                long_is_opened = True
                k_arr[i] = 1.
                price_prev = _p_arr[i]
                dg_arr[i] = 0
                di_arr[i+1] = -1.
            elif di_arr[i] * mu_arr[i] < 0.:
                des_str = 'Open short'
                short_is_opened = True
                k_arr[i] = -1.
                history = []
                price_prev = _p_arr[i]
                dg_arr[i] = 0
                di_arr[i+1] = 1.
            else:
                des_str = 'Hold'
                k_arr[i] = 0.
                dg_arr[i] = 0.
                di_arr[i+1] = 1.
                di_arr[i] = 0.
        if output_flag:
            print(f'iter={i}, t={round(t_arr[i]/3600., 2)} p={_p_arr[i]} mu={round(mu_arr[i], 4)}',
                  f'dmu={round(dmu_arr[i], 4)} dI={di_arr[i]} dg={dg_arr[i]}, '
                  f'K={round(k_arr[i], 4)} K_i={round(k_i_arr[i], 4)} -> {des_str}')
    t_ticks_arr = t_arr/3600.
    p_ser = pd.Series(_p_arr, index=t_ticks_arr)
    mu_ser = pd.Series(mu_arr, index=t_ticks_arr)
    dmu_ser = pd.Series(dmu_arr, index=t_ticks_arr)
    k_ser = pd.Series(k_arr, index=t_ticks_arr)
    k_i_ser = pd.Series(k_i_arr, index=t_ticks_arr)
    k_d_ser = pd.Series(k_d_arr, index=t_ticks_arr)
    k_dd_ser = pd.Series(k_dd_arr, index=t_ticks_arr)
    dg_ser = pd.Series(dg_arr, index=t_ticks_arr)
    di_ser = pd.Series(di_arr, index=t_ticks_arr)
    i_ser = pd.Series(np.cumsum(di_ser), index=t_ticks_arr)
    profit_ser = pd.Series(np.cumsum(dg_ser), index=t_ticks_arr)
    profit = profit_ser.iloc[-1]
    # print('Заработано:', profit)
    return p_ser, mu_ser, dmu_ser, dg_ser, profit_ser, di_ser, i_ser, k_ser, k_i_ser, k_d_ser, k_dd_ser


def visualize(p_ser, mu_ser, dmu_ser, dg_ser, profit_ser, di_ser, i_ser, k_ser, k_i_ser, k_d_ser, k_dd_ser):
    # fig, ((ax_p, ax_dg, ax_di, ax_k), (ax_mu, ax_profit, ax_i, ax_k)) = plt.subplots(figsize=(20, 3))
    n_max = len(p_ser)
    fig, ax = plt.subplots(figsize=(18, 10), nrows=4, ncols=3)
    plt.subplots_adjust(wspace=.2, hspace=.6)
    ax[0][0].set_title("Price")
    ax[0][0].set_xlabel("t")  # ось абсцисс
    ax[0][0].set_ylabel("p")  # ось ординат
    ax[0][0].plot(p_ser)
    ax[0][0].grid()


    ax[1][0].set_title("Trend")
    ax[1][0].set_xlabel("t")  # ось абсцисс
    ax[1][0].set_ylabel("mu")  # ось ординат
    ax[1][0].plot(mu_ser)
    ax[1][0].grid()

    ax[2][0].set_title("Trend derivarive")
    ax[2][0].set_xlabel("t")  # ось абсцисс
    ax[2][0].set_ylabel("mu'")  # ось ординат
    ax[2][0].plot(dmu_ser)
    ax[2][0].grid()

    ax[3][0].set_title("p'''")
    ax[3][0].set_xlabel("t")  # ось абсцисс
    ax[3][0].set_ylabel("p'''")  # ось ординат
    # ax[3][0].plot(p3_ser)
    ax[3][0].grid()


    ax[0][1].set_title("Instant profit")
    ax[0][1].set_xlabel("t")  # ось абсцисс
    ax[0][1].set_ylabel("dg")  # ось ординат
    ax[0][1].plot(dg_ser)
    ax[0][1].grid()

    ax[1][1].set_title("Cumulative profit")
    ax[1][1].set_xlabel("t")  # ось абсцисс
    ax[1][1].set_ylabel("Profit")  # ось ординат
    ax[1][1].plot(profit_ser)
    ax[1][1].grid()

    ax[2][1].set_title("Instant investment")
    ax[2][1].set_xlabel("t")  # ось абсцисс
    ax[2][1].set_ylabel("dI")  # ось ординат
    ax[2][1].plot(di_ser)
    ax[2][1].grid()

    ax[3][1].set_title("Cumulative investment")
    ax[3][1].set_xlabel("t")  # ось абсцисс
    ax[3][1].set_ylabel("I")  # ось ординат
    ax[3][1].plot(i_ser)
    ax[3][1].grid()

    ax[0][2].set_title("K proportional")
    ax[0][2].set_xlabel("t")  # ось абсцисс
    ax[0][2].set_ylabel("K")  # ось ординат
    ax[0][2].plot(k_ser)
    ax[0][2].grid()

    ax[1][2].set_title("K integral")
    ax[1][2].set_xlabel("t")  # ось абсцисс
    ax[1][2].set_ylabel("K_i")  # ось ординат
    ax[1][2].plot(k_i_ser)
    ax[1][2].grid()

    ax[2][2].set_title("K differential")
    ax[2][2].set_xlabel("t")  # ось абсцисс
    ax[2][2].set_ylabel("K_d")  # ось ординат
    ax[2][2].plot(k_d_ser)
    ax[2][2].grid()

    ax[3][2].set_title("K double-diff")
    ax[3][2].set_xlabel("t")  # ось абсцисс
    ax[3][2].set_ylabel("K_dd")  # ось ординат
    ax[3][2].plot(k_dd_ser)
    ax[3][2].grid()
    plt.show()


if __name__ == '__main__':
    data_tuple = load_test_data()
    res_tpl = calc_alg5(data_tuple[0], output_flag=True)
    visualize(*res_tpl)







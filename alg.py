import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from math import sin
from random import random
import pandas as pd
import matplotlib.pyplot as plt


def func_1():
    """Функция из первой ячейки, подбирает АРМА-параметры к синусоиде со случайным шумом"""
    n_max = 102
    # Задаем ряд
    price_ser_src = pd.Series([sin(x) + 10*random() for x in range(1, n_max)])
    # p_ser = price_ser_src[::5]
    # Находим mu (тренд)
    # Простой способ -- первая разность
    mu_arr = np.array([price_ser_src[i]-price_ser_src[i-1] if i > 0 else 0. for i in range(len(price_ser_src)) ])
    # Находим параметры ARMA (подгоняем модель)
    model = ARIMA(price_ser_src, order=(4, 0, 4))
    model_fit = model.fit()
    print(model_fit.summary())
    # make prediction -- это пока вообще не трогаем
    # yhat = model_fit.predict(len(data), len(data))
    # print(yhat)
    # График
    plt.title("Временной ряд") # заголовок
    plt.xlabel("n") # ось абсцисс
    plt.ylabel("p_n") # ось ординат
    plt.grid()      # включение отображение сетки
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

    fig = plt.figure(figsize=(20, 10))
    n_lst = list(range(len(p_arr_arma)))
    plt.subplot(2, 5, 1)
    plt.title("Price modeling by ARMA(4, 4) series")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("p_n")  # ось ординат
    plt.grid()  # включение отображение сетки
    # plt.plot(p_ser)  # построение графика
    plt.plot(n_lst, p_arr_arma)
    plt.legend(['p_n'], loc="upper left")

    plt.subplot(2, 5, 6)
    plt.title("Trend modeling by first difference")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("mu")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(n_lst, mu_arr_arma, color='orange')
    plt.legend(['mu'], loc="upper left")

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
    fig.subplots_adjust(hspace=.2)
    plt.show()

    return p_arr_stable, mu_arr_stable, \
        p_arr_growth, mu_arr_growth, \
        p_arr_si_14, mu_arr_si_14, \
        p_arr_si_21, mu_arr_si_21


p_arr_arma, mu_arr_arma = create_arma44_ser()
load_test_data()


import math
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from scipy.stats import t

# cleared_arr - выделяет из массива подмассив непропущенных данных
def cleared_arr(a):
    b=list()
    for i in range(len(a)):
        if np.isnan(a[i,1])==False:
            b.append(a[i])
    return np.squeeze(b)

# Отрисовка графика
def plot_maker(X1, Y1, X2, Y2, title):
    plt.title(title)
    plt.scatter(X1, Y1, color='blue', label='known data')
    plt.scatter(X2, Y2, color='red', label='imputed data')
    plt.xlim([50, 200])
    plt.ylim([50, 200])
    plt.legend(loc='upper left')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.show()


# Гауссовский шум, нужен для Stochastic regression imputation
def gaussian_noise(x, mu=0, std=None, random_state=8):
    rs = np.random.RandomState(random_state)
    noise = rs.normal(mu, std, size=x.shape)
    return x + noise


# Создаем двумерное нормальное распределение и случайным образом определяем 73 пропуска
def data_maker(n=100):
    sigma = np.array([[625, 375], [375, 625]], dtype=np.float64)
    don = np.random.multivariate_normal(size=n, mean=[125, 125], cov=sigma)
    f = [1 for i in range(27)]
    g = [0 for i in range(73)]
    h = f + g
    random.shuffle(h)
    donmiss = don.copy()
    for i in range(len(donmiss)):
        if h[i] == 0:
            donmiss[i, 1] = np.nan
    return don, donmiss

# Выборка с возвращением
def the_sample(X,n):
    sample=list()
    inds=np.random.randint(0, len(X),n)
    for i in range(len(inds)):
        sample.append(X[inds[i]])
    sample=np.array(sample)
    return sample[~np.isnan(sample[:,1])]

#Генерация совершенно случайных данных
def MCAR(arr):
    f = [1 for i in range(27)]
    g = [0 for i in range(73)]
    na = 73
    h = f + g
    random.shuffle(h)
    for i in range(len(arr)):
        if h[i] == 0:
            arr[i, 1] = np.nan
    return arr, na

#Генерация случайных данных
def MAR(arr):
    na = 0
    for i in range(len(arr)):
        if arr[i, 0] <= 140:
            arr[i, 1] = None
            na += 1
    return arr, na

#Генерация неслучайных данных
def MNAR(arr):
    na = 0
    for i in range(len(arr)):
        if arr[i, 1] <= 140:
            arr[i, 1] = None
            na += 1
    return arr, na

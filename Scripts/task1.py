import numpy as np
from timeit import default_timer as timer
import time
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

const = np.random.randint(0, 1, 1)
right_bound = 10


def f_const(v):
    start_time = time.time()
    const = np.random.randint(0, 1, 1)
    end_time = time.time() - start_time
    return end_time


def f_sum(v):
    start_time = timer()
    sum = 0
    for i in range(len(v)):
        sum = sum + v[i]
    end_time = timer() - start_time
    return end_time


def f_product(v):
    start_time = timer()
    product = 1
    for i in range(len(v)):
        product = product * v[i]
    # product = np.product(v)
    end_time = timer() - start_time
    return end_time


def f_norm(v):
    start_time = time.time()
    # norm = LA.norm(v)
    sum = 0
    for i in range(len(v)):
        sum = sum + v[i] ** 2
    norm = np.sqrt(sum)
    end_time = time.time() - start_time
    return end_time


def f_polynomial_direct(v, x):
    start_time = timer()
    p_x = 0
    for k in range(1, len(v + 1)):
        p_x += v[k] * x ** (k - 1)
    # p_x = np.sum([v[k] * x ** (k - 1) for k in range(1, len(v + 1))])
    end_time = timer() - start_time
    return end_time


def f_polynomial_horner(v, x):
    start_time = timer()
    p_x = v[0]
    for i in range(1, len(v)):
        p_x = p_x * x + v[i]
    end_time = timer() - start_time
    return end_time


def f_bubblesort(v):
    start_time = timer()
    for i in range(len(v) - 2):
        for j in range(len(v) - 2):
            if v[j] > v[j + 1]:
                v[j], v[j + 1] = v[j + 1], v[j]
    end_time = timer() - start_time
    return end_time


def matrix_product(n):
    start_time = timer()
    A = np.random.randint(right_bound, size=(n, n))
    B = np.random.randint(right_bound, size=(n, n))
    # A_dot_B = np.matmul(A, B)
    res = np.zeros((n,n))
    for i in range(len(B)):
        for j in range(len(B)):
            for k in range(len(B)):
                res[i][j] += A[i][k] * B[k][j]
    end_time = timer() - start_time
    return end_time


def plot_graph(empirical, n_list):
    sns.regplot(x=n_list, y=empirical)  # order=2
    plt.xlabel('n-dimension')
    plt.ylabel('time (sec)')
    plt.show()
    plt.plot(n_list, empirical)
    plt.xlabel('n-dimension')
    plt.ylabel('time (sec)')
    plt.show()


def plot_comparing_polinomials(time_polynomial_direct, time_polynomial_horner, n_list):
    direct = pd.DataFrame({'x1': n_list, 'y1': time_polynomial_direct})
    horner = pd.DataFrame({'x2': n_list, 'y2': time_polynomial_horner})
    direct_vs_horner = pd.concat([direct.rename(columns={'x1': 'n-dimension', 'y1': 'time(sec)'})
                                 .join(pd.Series(['direct'] * len(direct), name='direct_vs_horner')),
                                  horner.rename(columns={'x2': 'n-dimension', 'y2': 'time(sec)'})
                                 .join(pd.Series(['horner'] * len(horner), name='direct_vs_horner'))],
                                 ignore_index=True)
    pal = dict(direct="red", horner="blue")
    g = sns.FacetGrid(direct_vs_horner, hue='direct_vs_horner', palette=pal, size=7)
    g.map(sns.regplot, "n-dimension", "time(sec)", ci=None, robust=1)
    g.add_legend()
    plt.show()


def main():
    lim = 500
    time_const = np.zeros(100)
    # time_sum = np.zeros(lim // 50)
    # time_product = np.zeros(lim // 50)
    # time_norm = np.zeros(lim // 50)
    # time_polynomial_direct = np.zeros(lim // 50)
    # time_polynomial_horner = np.zeros(lim // 50)
    # time_bubblesort = np.zeros(lim // 50)
    time_matrix_product = np.zeros(lim // 50)
    n_list = [i for i in range(0, lim, 50)]
    n_list[0] = 1
    for n in n_list:
        vector = np.random.uniform(1, 10, n)
        # time_const_temp = np.zeros(5)
        # time_sum_temp = np.zeros(5)
        # time_product_temp = np.zeros(5)
        # time_norm_temp = np.zeros(5)
        time_polynomial_direct_temp = np.zeros(5)
        time_polynomial_horner_temp = np.zeros(5)
        # time_bubblesort_temp = np.zeros(5)
        time_matrix_product_temp = np.zeros(5)
        for k in range(5):
            # time_const_temp[k] = f_const(vector)
            # time_sum_temp[k] = f_sum(vector)
            # time_product_temp[k] = f_product(vector)
            # time_norm_temp[k] = f_norm(vector)
            # time_polynomial_direct_temp[k] = f_polynomial_direct(vector, 1.5)
            # time_polynomial_horner_temp[k] = f_polynomial_horner(vector, 1.5)
            # time_bubblesort_temp[k] = f_bubblesort(vector)
            time_matrix_product_temp[k] = matrix_product(n)
        # time_const[n // 50] = np.average(time_const_temp)
        # time_sum[n // 50] = np.average(time_sum_temp)
        # time_product[n // 50] = np.average(time_product_temp)
        # time_norm[n // 50] = np.average(time_norm_temp)
        # time_polynomial_direct[n // 50] = np.average(time_polynomial_direct_temp)
        # time_polynomial_horner[n // 50] = np.average(time_polynomial_horner_temp)
        # time_bubblesort[n // 50] = np.average(time_bubblesort_temp)
        time_matrix_product[n // 50] = np.average(time_matrix_product_temp)
    # plot_graph(time_const, n_list)
    # plot_graph(time_sum, n_list)
    # plot_graph(time_product, n_list)
    # plot_graph(time_norm, n_list)
    # plot_graph(time_polynomial_direct, n_list)
    # plot_graph(time_polynomial_horner, n_list)
    # plot_comparing_polinomials(time_polynomial_direct, time_polynomial_horner, n_list)
    # plot_graph(time_bubblesort, n_list)
    plot_graph(time_matrix_product, n_list)


if __name__ == '__main__':
    main()

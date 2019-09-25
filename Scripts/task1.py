import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
from scipy import stats

const = np.random.randint(0, 1, 1)  # const for const function
n_bound = 550  # dimension of a vector
k_iterations = 5  # number of runs for each experiment
right_upper_bound = 10  # right upper bound for sampling from uniform distribution
x = 1.5  # value of parameter x for polynomial evaluation
step = 50  # step for choosing n-dimension

# initialization of arrays containing average time for each n (n given with step 50)
time_const = np.zeros(n_bound // step)
time_sum = np.zeros(n_bound // step)
time_product = np.zeros(n_bound // step)
time_norm = np.zeros(n_bound // step)
time_polynomial_direct = np.zeros(n_bound // step)
time_polynomial_horner = np.zeros(n_bound // step)
time_bubblesort = np.zeros(n_bound // step)
time_matrix_product = np.zeros(n_bound // step)

n_list = [i for i in range(0, n_bound, step)]  # list of all dimensions
n_list[0] = 1


def get_time(function, vec):
    start_time = timer()
    _ = function(vec)
    end_time = timer() - start_time
    return end_time


def f_const(v):
    return const


def f_sum(v):
    sum = 0
    for i in range(len(v)):
        sum = sum + v[i]
    return sum


def f_product(v):
    product = 1
    for i in range(len(v)):
        product = product * v[i]
    return product


def f_norm(v):
    sum = 0
    for i in range(len(v)):
        sum = sum + v[i] ** 2
    norm = np.sqrt(sum)
    return norm


def f_polynomial_direct(v):
    p_x = 0
    for k in range(1, len(v) + 1):
        power_of_x = 1
        for _ in range(1, k):
            power_of_x = power_of_x * x
        p_x += v[k - 1] * power_of_x
    return p_x


def f_polynomial_horner(v):
    p_x = v[0]
    for i in range(1, len(v)):
        p_x = p_x * x + v[i]
    return p_x


def f_bubblesort(v):
    for i in range(len(v) - 2):
        for j in range(len(v) - 2):
            if v[j] > v[j + 1]:
                v[j], v[j + 1] = v[j + 1], v[j]
    return v


def matrix_product(n):
    A = [[random.randrange(10) for _ in range(n)] for _ in range(n)]
    B = [[random.randrange(10) for _ in range(n)] for _ in range(n)]
    product = [[0 for _ in range(len(A))] for _ in range(len(A))]
    for i in range(len(B)):
        for j in range(len(B)):
            for k in range(len(B)):
                product[i][j] += A[i][k] * B[k][j]
    return product


def plot_graph(empirical):
    plt.plot(n_list, empirical)  # plot just empirical
    plt.xlabel('n-dimension')
    plt.ylabel('time (sec)')
    plt.show()
    # order parameter was varied for each example
    sns.regplot(x=n_list, y=empirical, order=3)  # plot empirical approximated by theoretical
    plt.xlabel('n-dimension')
    plt.ylabel('time (sec)')
    plt.show()


def quantile_plot(x, y, **kwargs):
    _, xr = stats.probplot(x, fit=False)
    _, yr = stats.probplot(y, fit=False)
    plt.scatter(xr, yr, **kwargs)


def plot_comparing_polynomials(time_polynomial_direct, time_polynomial_horner):
    direct = pd.DataFrame({'x1': n_list, 'y1': time_polynomial_direct})
    horner = pd.DataFrame({'x2': n_list, 'y2': time_polynomial_horner})
    direct_vs_horner = pd.concat([direct.rename(columns={'x1': 'n-dimension', 'y1': 'time(sec)'})
                                 .join(pd.Series(['direct'] * len(direct), name='direct_vs_horner')),
                                  horner.rename(columns={'x2': 'n-dimension', 'y2': 'time(sec)'})
                                 .join(pd.Series(['horner'] * len(horner), name='direct_vs_horner'))],
                                 ignore_index=True)
    pal = dict(direct="red", horner="blue")
    g = sns.FacetGrid(direct_vs_horner, hue='direct_vs_horner', palette=pal, size=7)
    g.map(quantile_plot, "n-dimension", "time(sec)")
    g.add_legend()
    plt.show()


def main():
    for n in n_list:
        # generate vector of size n taken from uniform distribution [1,right_upper_bound)
        vector = np.random.uniform(1, right_upper_bound, n)
        # initialize arrays storing time for each run for a particular n
        time_const_temp = np.zeros(k_iterations)
        time_sum_temp = np.zeros(k_iterations)
        time_product_temp = np.zeros(k_iterations)
        time_norm_temp = np.zeros(k_iterations)
        time_polynomial_direct_temp = np.zeros(k_iterations)
        time_polynomial_horner_temp = np.zeros(k_iterations)
        time_bubblesort_temp = np.zeros(k_iterations)
        time_matrix_product_temp = np.zeros(k_iterations)
        for k in range(k_iterations):
            time_const_temp[k] = get_time(f_const, vector)
            time_sum_temp[k] = get_time(f_sum, vector)
            time_product_temp[k] = get_time(f_product, vector)
            time_norm_temp[k] = get_time(f_norm, vector)
            time_polynomial_direct_temp[k] = get_time(f_polynomial_direct, vector)
            time_polynomial_horner_temp[k] = get_time(f_polynomial_horner, vector)
            time_bubblesort_temp[k] = get_time(f_bubblesort, vector)
            time_matrix_product_temp[k] = get_time(matrix_product, n)
        time_const[n // step] = np.average(time_const_temp)
        time_sum[n // step] = np.average(time_sum_temp)
        time_product[n // step] = np.average(time_product_temp)
        time_norm[n // step] = np.average(time_norm_temp)
        time_polynomial_direct[n // step] = np.average(time_polynomial_direct_temp)
        time_polynomial_horner[n // step] = np.average(time_polynomial_horner_temp)
        time_bubblesort[n // step] = np.average(time_bubblesort_temp)
        time_matrix_product[n // step] = np.average(time_matrix_product_temp)
    plot_graph(time_const)
    plot_graph(time_sum)
    plot_graph(time_product)
    plot_graph(time_norm)
    plot_graph(time_polynomial_horner)
    plot_comparing_polynomials(time_polynomial_direct, time_polynomial_horner)
    plot_graph(time_bubblesort)
    plot_graph(time_matrix_product)


if __name__ == '__main__':
    main()

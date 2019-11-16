import math
import numpy as np
import matplotlib.pyplot as plt

epsilon = 0.001
gd_epsilon = 0.0001
alpha = np.random.rand(1)
betta = np.random.rand(1)


def generate_dataset():
    dataset = []
    x = []
    y = []
    for k in range(0, 1001):
        x_k = k / 1000
        y_k = alpha * x_k + betta + np.random.normal(0, 1, 1)[0]
        x.append(x_k)
        y.append(y_k)
    dataset.append(x)
    dataset.append(y)
    return dataset


dataset = generate_dataset()


def golden_section(functional, interval):
    a = interval[0]
    b = interval[1]
    gr = (3 - math.sqrt(5)) / 2
    gr2 = (-3 + math.sqrt(5)) / 2
    x_1 = a + gr * (b - a)
    x_2 = b + gr2 * (b - a)
    y_1 = functional(lr=x_1)
    y_2 = functional(lr=x_2)
    while not (math.fabs(a - b) < epsilon):
        if y_1 < y_2:
            b = x_2
            x_2 = x_1
            y_2 = y_1
            x_1 = a + gr * (b - a)
            y_1 = functional(lr=x_1)
        else:
            a = x_1
            x_1 = x_2
            y_1 = y_2
            x_2 = b + gr2 * (b - a)
            y_2 = functional(lr=x_2)
    return a, b


def plot_dataset(regression, optimal_gd, optimal_conjugate, optimal_newton, optimal_lm):
    x = dataset[0]
    result_optimal_gd = regression[1](np.array(optimal_gd[0]), np.array(optimal_gd[1]))
    result_optimal_conjugate = regression[1](np.array(optimal_conjugate[0]), np.array(optimal_conjugate[1]))
    result_optimal_newton = regression[1](np.array(optimal_newton[0]), np.array(optimal_newton[1]))
    result_optimal_lm = regression[1](np.array(optimal_lm[0]), np.array(optimal_lm[1]))
    plt.plot(x, dataset[1], 'o')
    plt.xlabel('x')
    plt.ylabel('y')
    if regression[0] == 'linear':
        initial = regression[1](alpha, betta)
        plt.plot(x, initial, label='initial')
    plt.plot(x, result_optimal_gd, label='gradient descent')
    plt.plot(x, result_optimal_conjugate, label='conjugate gradient descent')
    plt.plot(x, result_optimal_newton, label='Newton method')
    plt.plot(x, result_optimal_lm, label='Levenberg Marquardt')
    plt.legend(framealpha=1, frameon=True)
    plt.title('plot of {} regression'.format(regression[0]))
    plt.show()


def plot_functional_history(iterations, functional_history):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_ylabel('Value of functional')
    ax.set_xlabel('Iterations')
    _ = ax.plot(range(iterations), np.array(functional_history), 'b.')
    plt.show()

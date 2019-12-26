import random

import matplotlib.pyplot as plt
import numpy as np

epsilon = 0.001
n = 1000
maxiter = 1000

function = lambda x: 1 / (x ** 2 - 3 * x + 2)

# generate dataset
x_s = np.array([3 * k / n for k in range(n + 1)])
y_s = np.array([random.normalvariate(0, 1) for _ in range(n + 1)]) + np.array(
    [-100 if function(x) < -100 else function(x) if function(x) <= 100 else 100 for x in x_s])


def plot_dataset(regression, optimal_nm, optimal_lm, optimal_de):
    result_optimal_nm = regression(np.array(optimal_nm[0]), np.array(optimal_nm[1]), np.array(optimal_nm[2]),
                                   np.array(optimal_nm[3]))
    result_optimal_lm = regression(np.array(optimal_lm[0]), np.array(optimal_lm[1]), np.array(optimal_lm[2]),
                                   np.array(optimal_lm[3]))
    result_optimal_sa = regression(np.array(optimal_de[0]), np.array(optimal_de[1]), np.array(optimal_de[2]),
                                   np.array(optimal_de[3]))
    plt.plot(x_s, y_s, 'o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x_s, result_optimal_nm, label='Nelder-Mead')
    plt.plot(x_s, result_optimal_lm, label='Levenberg Marquardt')
    plt.plot(x_s, result_optimal_sa, label='Differential evolution')
    plt.legend(framealpha=1, frameon=True)
    plt.title('plot of rational regression')
    plt.show()

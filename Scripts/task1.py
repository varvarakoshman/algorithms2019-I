import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
import scipy.misc as spm
import functools as fun

alpha = np.random.rand(1)
betta = np.random.rand(1)
epsilon = 0.001

linear_reg_f = lambda x, a, b: a * x + b
rational_reg_f = lambda x, a, b: a / (1 + b * x)
stop_condition = lambda previous, current: np.linalg.norm(previous - current) < epsilon


def gradient_descent(regression):
    previous = np.array([np.random.rand(1)[0], np.random.rand(1)[0]])
    step = 0.001
    iterations = 0
    while True:
        current = previous - step * np.gradient(regression, previous)
        if stop_condition(previous, current):
            break
        else:
            previous = current
            iterations += 1
    return current, iterations


def conjugate_gradient_method(regression):
    return None, None


def newton_method(regression):
    return None, None


def levenberg_marquardt_algorithm(regression):
    return None, None


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


def plot_dataset(regression, optimal_gd, optimal_conjugate, optimal_newton, optimal_lm):
    x = dataset[0]
    initial = regression[1](x, alpha, betta)
    result_optimal_gd = regression[1](x, np.array(optimal_gd[0]), np.array(optimal_gd[1]))
    result_optimal_conjugate = regression[1](x, np.array(optimal_conjugate[0]), np.array(optimal_conjugate[1]))
    result_optimal_newton = regression[1](x, np.array(optimal_newton[0]), np.array(optimal_newton[1]))
    result_optimal_lm = regression[1](x, np.array(optimal_lm[0]), np.array(optimal_lm[1]))
    plt.plot(x, dataset[1], 'o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, initial, label='initial')
    plt.plot(x, result_optimal_gd, label='gradient descent')
    plt.plot(x, result_optimal_conjugate, label='conjugate gradient descent')
    plt.plot(x, result_optimal_newton, label='Newton method')
    plt.plot(x, result_optimal_lm, label='Levenberg Marquardt')
    plt.legend(framealpha=1, frameon=True)
    plt.title('plot of {} regression'.format(regression[0]))
    plt.show()


def functional(arg, regression):
    functional_value = 0
    for i in range(len(dataset[0])):
        x_k = dataset[0][i]
        y_k = dataset[1][i][0]
        functional_value += np.square(regression(x_k, arg[0], arg[1]) - y_k)
    return functional_value


dataset = generate_dataset()


def main():
    regressions = [('linear', linear_reg_f), ('rational', rational_reg_f)]
    for regression in regressions:
        optimal_gd, gd_iterations = gradient_descent(regression[1])
        optimal_conjugate, con_iterations = conjugate_gradient_method(regression[1])
        optimal_newton, newton_iterations = newton_method(regression[1])
        optimal_lm, lm_iterations = levenberg_marquardt_algorithm(regression[1])
        plot_dataset(regression, optimal_gd, optimal_conjugate, optimal_newton, optimal_lm)
        print("coefficients initial: ", alpha[0], betta[0])
        print("coefficients with Gradient Descent method (for {} regression) ({} iterations): "
              .format(regression[0], gd_iterations), optimal_gd[0], optimal_gd[1])
        print("coefficients with Conjugate Gradient Descent method (for {} regression) ({} iterations): "
              .format(regression[0], con_iterations), optimal_conjugate[0], optimal_conjugate[1])
        print("coefficients with Newton method (for {} regression) ({} iterations): "
              .format(regression[0], newton_iterations), optimal_newton[0], optimal_newton[1])
        print("coefficients with Levenberg-Marquardt algorithm (for {} regression) ({} iterations): "
              .format(regression[0], lm_iterations), optimal_lm[0], optimal_lm[1])


if __name__ == '__main__':
    main()

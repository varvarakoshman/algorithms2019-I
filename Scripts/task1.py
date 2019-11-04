import numpy as np
import scipy.optimize as sp
import functools as fun
from mpmath import *

from Scripts.utility import golden_section, epsilon, dataset, alpha, betta

linear_reg_f = lambda x, a, b: a * x + b
rational_reg_f = lambda x, a, b: a / (1 + b * x)
stop_condition = lambda previous, current: np.linalg.norm(previous - current) < epsilon
functional_linear = lambda a, b: np.sum(
    np.array([np.square(a * dataset[0][i] + b - dataset[1][i][0]) for i in range(len(dataset[0]))]))
functional_rational = lambda a, b: np.sum(
    np.array([np.square(a / (1 + b * dataset[0][i]) - dataset[1][i][0]) for i in range(len(dataset[0]))]))
update_function = lambda param, dfdparam, lr: param - lr * dfdparam


def steepest_gradient_descent(functional):
    previous = np.array([np.random.rand(1)[0], np.random.rand(1)[0]])  # starting point
    iterations = 0
    while True:
        df_da = diff(functional, (previous[0], previous[1]), (1, 0))
        df_db = diff(functional, (previous[0], previous[1]), (0, 1))
        gradient = np.array([df_da, df_db])
        left, right = golden_section(fun.partial(update_function, param=previous[0], dfdparam=df_da), (0.0001, 0.1))
        optimal_step_a = 0.5 * (left + right)
        left, right = golden_section(fun.partial(update_function, param=previous[1], dfdparam=df_db), (0.0001, 0.1))
        optimal_step_b = 0.5 * (left + right)
        optimal_steps = np.array([optimal_step_a, optimal_step_b])
        current = previous - optimal_steps * gradient
        iterations += 1
        if stop_condition(previous, current):
            break
        else:
            previous = current
    return current, iterations


def gradient_descent(functional):
    previous = np.array([np.random.rand(1)[0], np.random.rand(1)[0]])  # starting point
    iterations = 0
    lr = 0.005
    while True:
        df_da = diff(functional, (previous[0], previous[1]), (1, 0))
        df_db = diff(functional, (previous[0], previous[1]), (0, 1))
        gradient = np.array([df_da, df_db])
        current = previous - lr * gradient
        iterations += 1
        if stop_condition(previous, current):
            break
        else:
            previous = current
    return current, iterations


def newton_method(functional):
    return None, None


def levenberg_marquardt_algorithm(functional):
    return None, None


def main():
    functionals = [functional_linear, functional_rational]
    regressions = [('linear', linear_reg_f), ('rational', rational_reg_f)]
    for i in range(len(functionals)):
        optimal_gd, gd_iterations = steepest_gradient_descent(functionals[i])
        # optimal_conjugate, con_iterations = conjugate_gradient_method(functionals[i])
        python_nm_res = sp.minimize(functionals[i],
                                    np.array((np.random.rand(1)[0], np.random.rand(1)[0])),
                                    method="CG",
                                    options={'gtol': 1e-3})
        optimal_newton, newton_iterations = newton_method(functionals[i])
        optimal_lm, lm_iterations = levenberg_marquardt_algorithm(functionals[i])
        # plot_dataset(regressions[i], optimal_gd, optimal_conjugate, optimal_newton, optimal_lm)
        print("coefficients initial: ", alpha[0], betta[0])
        # print("coefficients with Gradient Descent method (for {} regression) ({} iterations): "
        #       .format(regressions[i][0], gd_iterations), optimal_gd[0], optimal_gd[1])
        # print("coefficients with Conjugate Gradient Descent method (for {} regression) ({} iterations): "
        #       .format(regressions[i][0], con_iterations), optimal_conjugate[0], optimal_conjugate[1])
        # print("coefficients with Newton method (for {} regression) ({} iterations): "
        #       .format(regressions[i][0], newton_iterations), optimal_newton[0], optimal_newton[1])
        # print("coefficients with Levenberg-Marquardt algorithm (for {} regression) ({} iterations): "
        #       .format(regressions[i][0], lm_iterations), optimal_lm[0], optimal_lm[1])
        print()


if __name__ == '__main__':
    main()

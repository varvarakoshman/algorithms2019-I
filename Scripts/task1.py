import numpy as np
import scipy.optimize as sp
import functools as fun
from mpmath import *
from numpy import linalg
from scipy.optimize import least_squares

from Scripts.utility import golden_section, epsilon, dataset, alpha, betta, gd_epsilon, plot_dataset, \
    plot_functional_history

linear_reg_f = lambda a, b: np.array([a * dataset[0][i] + b for i in range(len(dataset[0]))])
rational_reg_f = lambda a, b: np.array([a / (1 + b * dataset[0][i]) for i in range(len(dataset[0]))])
stop_condition = lambda previous, current: np.linalg.norm(previous - current) < epsilon
functional_linear = lambda point: np.sum(
    np.array([np.square(point[0] * dataset[0][i] + point[1] - dataset[1][i][0]) for i in range(len(dataset[0]))]))
functional_rational = lambda point: np.sum(
    np.array([np.square(point[0] / (1 + point[1] * dataset[0][i]) - dataset[1][i][0]) for i in range(len(dataset[0]))]))
update_function = lambda param, dfdparam, lr: param - lr * dfdparam

# with 2 parameters for taking partial derivatives
functional_linear2 = lambda a, b: np.sum(
    np.array([np.square(a * dataset[0][i] + b - dataset[1][i][0]) for i in range(len(dataset[0]))]))
functional_rational2 = lambda a, b: np.sum(
    np.array([np.square(a / (1 + b * dataset[0][i]) - dataset[1][i][0]) for i in range(len(dataset[0]))]))


def steepest_gradient_descent(functional):
    prev = np.array([np.random.rand(1)[0], np.random.rand(1)[0]])  # starting point
    iterations = 0
    functional_history = []
    while True:
        df_da = diff(functional, (prev[0], prev[1]), (1, 0))
        df_db = diff(functional, (prev[0], prev[1]), (0, 1))
        gradient = np.array([df_da, df_db])

        left1, right1 = golden_section(fun.partial(update_function, param=prev[0], dfdparam=df_da), (0.00001, 0.001))
        optimal_step_a = 0.5 * (left1 + right1)
        left2, right2 = golden_section(fun.partial(update_function, param=prev[1], dfdparam=df_db), (0.00001, 0.001))
        optimal_step_b = 0.5 * (left2 + right2)
        optimal_steps = np.array([optimal_step_a, optimal_step_b])

        curr = prev - optimal_steps * gradient
        iterations += 1
        functional_history.append(functional(curr[0], curr[1]))
        if stop_condition(prev, curr):
            break
        else:
            prev = curr
    # plot_functional_history(iterations, functional_history)
    return curr, iterations


def gradient_descent(functional):
    prev = np.array([np.random.rand(1)[0], np.random.rand(1)[0]])  # starting point
    iterations = 0
    functional_history = []
    lr = 0.0001
    while True:
        df_da = float(diff(functional, (prev[0], prev[1]), (1, 0)))
        df_db = float(diff(functional, (prev[0], prev[1]), (0, 1)))
        gradient = np.array([df_da, df_db])
        curr = prev - lr * gradient
        iterations += 1
        functional_history.append(functional(curr[0], curr[1]))
        if stop_condition(prev, curr):
            break
        else:
            prev = curr
    # plot_functional_history(iterations, functional_history)
    return curr, iterations


def newton_method(functional):
    prev = np.array([np.random.rand(1)[0], np.random.rand(1)[0]])  # starting point
    iterations = 0
    while True:
        df_da = diff(functional, (prev[0], prev[1]), (1, 0))
        df_db = diff(functional, (prev[0], prev[1]), (0, 1))
        gradient_f = np.array([df_da, df_db])
        hessian_f = None  # haven't found any normal way to compute it (maybe go and use pytorch (haha classic))
        curr = prev - hessian_f ** (-1) * gradient_f
        iterations += 1
        if stop_condition(prev, curr):
            break
        else:
            prev = curr
    return curr, iterations


def levenberg_marquardt_algorithm(functional, regression):
    prev = np.array([np.random.rand(1)[0], np.random.rand(1)[0]])  # starting point
    iterations = 0
    lr = np.random.rand(1)[0]
    factor = 2  # damping factor should be > 1
    k = 1  # "some k" according to wiki
    while True:
        while True:
            df_da = diff(regression, (prev[0], prev[1]), (1, 0))
            df_db = diff(regression, (prev[0], prev[1]), (0, 1))
            gradient_f = np.array(
                [[float(df_da[i]) for i in range(len(df_da))], [float(df_db[i]) for i in range(len(df_da))]])
            j_jT = gradient_f.dot(gradient_f.transpose())
            H_f = j_jT + lr * np.ones((len(j_jT), len(j_jT)))
            f_s = regression(np.array(prev[0]), np.array(prev[1]))
            y = np.concatenate(dataset[1], axis=0)
            error = gradient_f.dot(y - f_s)
            delta = linalg.inv(H_f).dot(error)
            curr = prev + delta
            iterations += 1
            if functional(curr) < functional(prev):
                break
            else:
                lr = lr * factor ** k
        if stop_condition(prev, curr):
            break
        else:
            prev = curr
    return curr, iterations


def main():
    functionals = [(functional_linear, functional_linear2), (functional_rational, functional_linear2)]
    regressions = [('linear', linear_reg_f), ('rational', rational_reg_f)]
    jac = lambda point: np.array([float(diff(functionals[i][1], (point[0], point[1]), (1, 0))),
                                  float(diff(functionals[i][1], (point[0], point[1]), (0, 1)))])
    for i in range(len(functionals)):
        optimal_gd, gd_iterations = steepest_gradient_descent(functionals[i][1])
        # optimal_gd, gd_iterations = gradient_descent(functionals[i][1]) # долго сходится
        python_cg_res = sp.minimize(functionals[i][0],
                                    np.array((np.random.rand(1)[0], np.random.rand(1)[0])),
                                    method="CG",
                                    options={'gtol': epsilon})
        optimal_cg = python_cg_res.x
        cg_iterations = python_cg_res.nit
        # optimal_newton, newton_iterations = newton_method(functionals[i])
        newton_res = sp.minimize(functionals[i][0],
                                 np.array((np.random.rand(1)[0], np.random.rand(1)[0])),
                                 jac=jac,
                                 method='Newton-CG',
                                 options={'gtol': epsilon})
        optimal_newton = newton_res.x
        newton_iterations = newton_res.nit
        optimal_lm, lm_iterations = levenberg_marquardt_algorithm(functionals[i][0], regressions[i][1])
        # python_lm_res = sp.least_squares(functionals[i][0],
        #                                  x0=np.array((np.random.rand(1)[0], np.random.rand(1)[0])),
        #                                  jac=jac,
        #                                  method="lm",
        #                                  xtol=epsilon)
        # - не работает, ибо почему-то компилятор воспринимает размерность x0 как 2, а не 1 (хотя это кортеж),
        #  падает, когда пытается эту пару аргументов отправить на вход функционалу, который принимает 1 аргумент.
        plot_dataset(regressions[i], optimal_gd, optimal_cg, optimal_newton, optimal_lm)
        print("coefficients initial: ", alpha[0], betta[0])
        print("coefficients with Gradient Descent method (for {} regression) ({} iterations): "
              .format(regressions[i][0], gd_iterations), optimal_gd[0], optimal_gd[1])
        print("coefficients with Conjugate Gradient Descent method (for {} regression) ({} iterations): "
              .format(regressions[i][0], cg_iterations), optimal_cg[0], optimal_cg[1])
        print("coefficients with Newton method (for {} regression) ({} iterations): "
              .format(regressions[i][0], newton_iterations), optimal_newton[0], optimal_newton[1])
        print("coefficients with Levenberg-Marquardt algorithm (for {} regression) ({} iterations): "
              .format(regressions[i][0], lm_iterations), optimal_lm[0], optimal_lm[1])
        print("functional value at found point (gd)", functionals[i][0](optimal_gd))
        print("functional value at found point (cg)", functionals[i][0](optimal_cg))
        print("functional value at found point (newton)", functionals[i][0](optimal_newton))
        print("functional value at found point (lm)", functionals[i][0](optimal_lm))
        print()


if __name__ == '__main__':
    main()

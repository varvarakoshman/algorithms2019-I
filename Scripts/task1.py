import numpy as np
import scipy.optimize as sp

from Scripts.utility import plot_dataset, epsilon, x_s, y_s, maxiter

rational_reg_f = lambda a, b, c, d: np.array((a * x_s + b) / (x_s * x_s + c * x_s + d))


def functional_rational(point):
    return np.sum(np.array([np.square((point[0] * x_s + point[1]) / (
            x_s * x_s + point[2] * x_s + point[3]) - y_s)]))


def helper_f(point):
    err = f(point) - y_s
    return err


def f(point):
    return (x_s * point[0] + point[1]) / (x_s * x_s + point[2] * x_s + point[3])


def main():
    x0 = np.array((np.random.rand(1)[0], np.random.rand(1)[0], np.random.rand(1)[0],
                   np.random.rand(1)[0]))  # initial guess

    nm_res = sp.minimize(functional_rational,
                         x0,
                         method="Nelder-Mead",
                         options={'xtol': 1e-3, 'ftol': 1e-3, 'maxiter': maxiter})
    nm_iter = nm_res.nit
    optimal_nm = nm_res.x
    print("coefficients with Nelder-Mead algorithm with ({} iterations): "
          .format(nm_iter), optimal_nm)
    print("functional value at found point (nm)", functional_rational(optimal_nm))
    print()

    lm_res = sp.least_squares(helper_f, x0,
                              method="lm",
                              max_nfev=maxiter,
                              xtol=epsilon)
    optimal_lm = lm_res.x
    lm_iterations = lm_res.nfev
    print("coefficients with Levenberg-Marquardt algorithm with ({} iterations): "
          .format(lm_iterations), optimal_lm)
    print("functional value at found point (lm)", functional_rational(optimal_lm))
    print()

    bounds = ((-3, 3), (-3, 3), (-3, 3), (-3, 3))
    res = sp.differential_evolution(functional_rational,
                                    bounds,
                                    maxiter=maxiter,
                                    tol=epsilon)
    optimal_de = res.x
    de_iterations = res.nit
    print("coefficients with Differential evolution algorithm with ({} iterations): "
          .format(de_iterations), optimal_de)
    print("functional value at found point (de)", functional_rational(optimal_de))
    print()

    plot_dataset(rational_reg_f, optimal_nm, optimal_lm, optimal_de)


if __name__ == '__main__':
    main()

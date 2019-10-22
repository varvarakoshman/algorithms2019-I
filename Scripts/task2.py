import numpy as np
import matplotlib.pyplot as plt

# task 2

alpha = np.random.normal(0, 1, 1)
betta = np.random.normal(0, 1, 1)
epsilon = 0.001

linear_reg_f = lambda x, a, b: a * x + b
rational_reg_f = lambda x, a, b: a / (1 + b * x)
stop_condition = lambda a_old, a_new, b_old, b_new: np.absolute(a_old - a_new) < epsilon \
                                                    and np.absolute(b_old - b_new) < epsilon


def minimize_f(dataset, regression_f, parameters):
    all_functional_values = {}
    changing = 0
    while changing < 1:
        functional_value = 0
        for i in range(len(dataset)):
            x_k = dataset[0][i]
            y_k = dataset[1][i]
            if parameters[0] is None:
                functional_value += np.square(regression_f(x_k, changing, parameters[1]) - y_k)[0]
            else:
                functional_value += np.square(regression_f(x_k, parameters[0], changing) - y_k)[0]
        all_functional_values.update({functional_value: changing})
        # all_functional_values[changing] = functional_value
        changing += epsilon
    optimal_parameter = all_functional_values.get(np.min(np.vstack(all_functional_values.keys())))
    return optimal_parameter


def gauss_method(dataset, regression_f):
    a_old = 0-epsilon
    b_old = 0-epsilon
    b_new = 0-epsilon
    while True:
        a_new = minimize_f(dataset, regression_f, (None, b_old))  # b fixed
        if stop_condition(a_old, a_new, b_old, b_new):
            break
        b_new = minimize_f(dataset, regression_f, (a_new, None))  # new a fixed
        if stop_condition(a_old, a_new, b_old, b_new):
            break
        else:
            a_old = a_new
            b_old = b_new
    return a_new, b_new


def plot_dataset(dataset):
    plt.plot(dataset[0], dataset[1], 'o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def generate_dataset():
    dataset = []
    x = []
    y = []
    for k in range(0, 101):
        x_k = k / 100
        y_k = alpha * x_k + betta + np.random.normal(0, 1, 1)
        x.append(x_k)
        y.append(y_k)
    dataset.append(x)
    dataset.append(y)
    return dataset


def main():
    dataset = generate_dataset()
    plot_dataset(dataset)
    optimal_a, optimal_b = gauss_method(dataset, linear_reg_f)
    print(alpha, betta)
    print(optimal_a, optimal_b)


if __name__ == '__main__':
    main()

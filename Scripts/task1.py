import math

# task 1

epsilon = 0.001
delta = 0.0005

f_1 = lambda x: x ** 3
f_2 = lambda x: math.fabs(x - 0.2)
f_3 = lambda x: x * math.sin(1 / x)

count_f_calls = 0
count_iterations = 0


def increment_f_calls():
    global count_f_calls
    count_f_calls += 1


def zero_f_calls():
    global count_f_calls
    count_f_calls = 0


def increment_iterations():
    global count_iterations
    count_iterations += 1


def zero_iterations():
    global count_iterations
    count_iterations = 0


def dichotomy(function, interval):
    a = interval[0]
    b = interval[1]
    while not (math.fabs(a - b) < epsilon):
        x_1 = (a + b - delta) / 2
        x_2 = (a + b + delta) / 2
        left_value = function(x_1)
        increment_f_calls()
        right_value = function(x_2)
        increment_f_calls()
        if left_value < right_value:
            b = x_1
        else:
            a = x_2
        increment_iterations()
    return a, b


def golden_section(function, interval):
    a = interval[0]
    b = interval[1]
    gr = (3 - math.sqrt(5)) / 2
    gr2 = (-3 + math.sqrt(5)) / 2
    x_1 = a + gr * (b - a)
    x_2 = b + gr2 * (b - a)
    y_1 = function(x_1)
    increment_f_calls()
    y_2 = function(x_2)
    increment_f_calls()
    increment_iterations()
    while not (math.fabs(a - b) < epsilon):
        if y_1 < y_2:
            b = x_2
            x_2 = x_1
            y_2 = y_1
            x_1 = a + gr * (b - a)
            y_1 = function(x_1)
            increment_f_calls()
        else:
            a = x_1
            x_1 = x_2
            y_1 = y_2
            x_2 = b + gr2 * (b - a)
            y_2 = function(x_2)
            increment_f_calls()
        increment_iterations()
    return a, b


def main():
    functions = [f_1, f_2, f_3]
    intervals = [[0, 1], [0, 1], [0.1, 1]]
    dichotomy_results = []
    gs_results = []
    for i in range(len(functions)):
        dichotomy_results.append(((dichotomy(functions[i], intervals[i])), count_f_calls, count_iterations))
        zero_f_calls()
        zero_iterations()
        gs_results.append(((golden_section(functions[i], intervals[i])), count_f_calls, count_iterations))
        zero_f_calls()
        zero_iterations()
    print("dichotomy: ", dichotomy_results)
    print("golden section: ", gs_results)


if __name__ == '__main__':
    main()

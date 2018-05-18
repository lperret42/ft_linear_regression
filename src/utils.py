import argparse
from pkg_resources import resource_filename
import csv
import numpy as np
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--show", action="store_true",
        help="display the data and the linear regression evolution on a graph")
    parser.add_argument("-sol", "--solution", action="store_true",
        help="compare the linear regression result with real solution")
    parser.add_argument("-n", "--nb_iter", action="store_true",
        help="print after how many iterations the algorithm has converged")
    parser.add_argument("-c", "--cost", action="store_true",
        help="print cost at each iterations of the algorithm")
    parser.add_argument('-l', '--learning_rate', help='must be float in ]0; 1]')
    args = parser.parse_args()

    return args

def get_data():
    params_file = resource_filename(__name__, '../data.csv')
    with open(params_file, 'r') as csvfile:
        lines  = [line for line in csv.reader(csvfile, delimiter=',')][1:]
        csvfile.close()

    lines = [list(map(float, line)) for line in lines]
    return lines

def linear_function(theta0, theta1, x):
    return theta0 + theta1 * x

def get_cost(theta0, theta1, X, Y):
    return sum([(y - (theta1 * x + theta0)) ** 2
            for x, y in zip(X, Y)]) /  (2* len(X))

def get_solution(X, Y):
    N = len(X)
    Sx = np.sum(X)
    Sy = np.sum(Y)
    Sxx = np.sum([x ** 2 for x in X])
    Sxy = np.sum([x * y for x, y in zip(X, Y)])
    theta1 = (N * Sxy - Sx * Sy) / (N * Sxx - Sx * Sx)
    theta0 = (Sy - theta1 * Sx) / N
    return [theta0, theta1]

def quit_figure(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)

def print_comparison(my_theta0, my_theta1, my_cost,
                                        real_theta0, real_theta1, real_cost):
    print("With linear regression:")
    print("Theta0:", my_theta0)
    print("Theta1:", my_theta1)
    print("Cost:", my_cost, "\n")
    print("With mathematical formula:")
    print("Theta0:", real_theta0)
    print("Theta01:", real_theta1)
    print("Cost:", real_cost, "\n")
    print("Mathematical cost is smaller than cost from",
                    "linear regression of {}".format(my_cost - real_cost))

def manage_learning_rate(learning_rate):
    if not (learning_rate is None or is_float(learning_rate)):
        return -1
    else:
        learning_rate = float(learning_rate) if learning_rate is not None else 1
    if not (learning_rate > 0 and learning_rate <= 1):
        return -1

    return learning_rate

def is_float(string):
    if len(string) == 0:
        return False
    if len(string) == 1:
        if not string[0] in "0123456789":
            return False
        else:
            return True
    if len(string) == 2 and (string == "-." or string == ".-"):
        return False
    if string.count('.') > 1:
        return False
    if '-' in string[1:]:
        return False
    for c in string:
        if not (c == '-' or c == '.' or c in "0123456789"):
            return False
    return True

def sqrt(x, epsilon=10e-15):
    """
    implementation of the sqrt function, both suites u and v converge to sqrt(x)
    """
    if x < 0:
        return None
    if x == 0:
        return 0
    u = 1
    v = x
    error_u = abs(u * u - x)
    error_v = abs(v * v - x)
    old_error_u = error_u
    old_error_v = error_v
    while error_u > epsilon and error_v > epsilon:
        tmp = u
        u = 2. / (1. / u + 1. / v)
        v = (tmp + v) / 2.
        error_u = abs(u * u - x)
        error_v = abs(v * v - x)
        if old_error_u == error_u and old_error_v == old_error_v:
            break
        old_error_u = error_u
        old_error_v = error_v

    return u if error_u <= error_v else v

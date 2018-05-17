#!/usr/bin/python

import argparse
from pkg_resources import resource_filename
import csv
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
import threading

from src.linear_function import LinearFunction
from src.linear_regressor import LinearRegressor

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--show", action="store_true",
        help="display the data and the linear regression result on a graph")
    args = parser.parse_args()

    return args

def get_data():
    params_file = resource_filename(__name__, 'data.csv')
    with open(params_file, 'r') as csvfile:
        lines  = [line for line in csv.reader(csvfile, delimiter=',')][1:]
        csvfile.close()

    lines = [list(map(float, line)) for line in lines]
    return lines

def main():
    args = parse_arguments()
    data = get_data()
    X = [x for x, y in data]
    Y = [y for x, y in data]
    linear_regressor = LinearRegressor(X, Y)
    if args.show:
        linear_function = LinearFunction(linear_regressor.theta0, linear_regressor.theta1)
        line_x = [min(linear_regressor.X), max(linear_regressor.X)]
        line_y = [linear_function.evaluate(i) for i in line_x]
        plt.plot(line_x, line_y, 'b')
        plt.plot(linear_regressor.X, linear_regressor.Y, 'ro'),
        t = threading.Thread(target=linear_regressor.train, kwargs={'max_iter':10e6,'show':True})
        t.start()
        plt.show()
    else:
        linear_regressor.train()

if __name__ == '__main__':
    main()

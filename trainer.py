#!/usr/bin/python

from pkg_resources import resource_filename
import csv
import matplotlib.pyplot as plt
import numpy as np

from src.linear_function import LinearFunction
from src.linear_regressor import LinearRegressor

def get_data():
    params_file = resource_filename(__name__, 'data.csv')
    with open(params_file, 'r') as csvfile:
        lines  = [line for line in csv.reader(csvfile, delimiter=',')][1:]
        csvfile.close()

    lines = [map(float, line) for line in lines]
    return lines

def main():
    data = get_data()
    X = [x for x, y in data]
    Y = [y for x, y in data]
    linear_regressor = LinearRegressor(X, Y)
    linear_regressor.train()
    linear_regressor.show()

if __name__ == '__main__':
    main()

#!/usr/bin/python

import argparse
from pkg_resources import resource_filename
import csv
import matplotlib.pyplot as plt
import numpy as np

from src.linear_function import LinearFunction
from src.linear_regressor import LinearRegressor

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-show', '--show', action='store_true',
        help='display the data and the linear regression result on a graph')
    args = parser.parse_args()

    return args

def get_data():
    params_file = resource_filename(__name__, 'data.csv')
    with open(params_file, 'r') as csvfile:
        lines  = [line for line in csv.reader(csvfile, delimiter=',')][1:]
        csvfile.close()

    lines = [map(float, line) for line in lines]
    return lines

def main():
    args = parse_arguments()
    data = get_data()
    X = [x for x, y in data]
    Y = [y for x, y in data]
    linear_regressor = LinearRegressor(X, Y)
    linear_regressor.train()
    if args.show:
        linear_regressor.show()

if __name__ == '__main__':
    main()

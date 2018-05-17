#!/usr/bin/python
# -*- coding: utf-8 -*-

from pkg_resources import resource_filename

import csv
import matplotlib.pyplot as plt
import numpy as np

from src.linear_function import LinearFunction

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

def get_params():
    params_file = resource_filename(__name__, 'params.csv')
    with open(params_file, 'r') as csvfile:
        lines  = [line for line in csv.reader(csvfile, delimiter=',')][1:]
        csvfile.close()
    theta0, theta1 = map(float, lines[0])

    return theta0, theta1

def loop_on_input():
    theta0, theta1 = get_params()
    linear_function = LinearFunction(theta0, theta1)
    while True:
        print("Please enter a mileage :")
        string = input()
        if string == "quit":
            break
        if not is_float(string):
            print("Mileage must be a float number :")
            continue
        nb = float(string)
        if nb <= 0:
            print("Mileage must be positive")
            continue
        price = linear_function.evaluate(nb)
        price = int(price)
        print("This car worth {} euros".format(price))

def main():
    loop_on_input()

if __name__ == '__main__':
    main()

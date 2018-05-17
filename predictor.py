#!/usr/bin/python
# -*- coding: utf-8 -*-

from pkg_resources import resource_filename

import csv
import matplotlib.pyplot as plt
import numpy as np

from src.linear_function import LinearFunction

def is_float(str):
    for c in str:
        if not(c == '.' or c in "0123456789"):
            return False
    return True

def get_params():
    params_file = resource_filename(__name__, 'params.csv')
    with open(params_file, 'r') as csvfile:
        lines  = [line for line in csv.reader(csvfile, delimiter=',')][1:]
        csvfile.close()
    theta0, theta1 = map(int, lines[0])

    return theta0, theta1

def loop_on_input():
    theta0, theta1 = get_params()
    linear_function = LinearFunction(theta0, theta1)
    while True:
        print "Please enter a mileage :"
        str = raw_input()
        if str == "quit":
            break
        if not is_float(str):
            print "Mileage must be a float number :"
            continue
        nb = float(str)
        price = linear_function.evaluate(nb)
        price = int(price) if int(price) == price else price
        print "This car worth {} euros".format(price)

def main():
    loop_on_input()

if __name__ == '__main__':
    main()

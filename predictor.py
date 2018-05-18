#!/Users/lperret/.brew/Cellar/python/3.6.5/bin/python3.6
# -*- coding: utf-8 -*-

from pkg_resources import resource_filename
import csv

from src.utils import linear_function, is_float

def get_params():
    params_file = resource_filename(__name__, 'params.csv')
    with open(params_file, 'r') as csvfile:
        lines  = [line for line in csv.reader(csvfile, delimiter=',')][1:]
        csvfile.close()
    theta0, theta1 = map(float, lines[0])

    return theta0, theta1

def loop_on_input():
    theta0, theta1 = get_params()
    while True:
        print("Please enter a mileage :")
        string = input()
        if string == "quit":
            break
        if not is_float(string):
            print("Mileage must be a float number :")
            continue
        nb = float(string)
        if nb < 0:
            print("Mileage must be positive")
            continue
        price = linear_function(theta0, theta1, nb)
        price = int(price)
        print("This car worth {} euros".format(price))

def main():
    loop_on_input()

if __name__ == '__main__':
    main()

#!/Users/lperret/.brew/Cellar/python/3.6.5/bin/python3.6

import time
import threading
import matplotlib.pyplot as plt

from src.linear_regressor import LinearRegressor
from src.utils import parse_arguments, get_data, linear_function, get_cost,\
    print_comparison, get_solution, quit_figure, is_float, manage_learning_rate

def main():
    args = parse_arguments()
    data = get_data()
    X, Y = [x for x, _ in data], [y for _, y in data]
    learning_rate = manage_learning_rate(args.learning_rate)
    if not (learning_rate > 0 and learning_rate <= 1):
        print("Learning rate must be a float in ]0;1]")
        return
    linear_regressor = LinearRegressor(X, Y, learning_rate=learning_rate)
    if args.show:
        line_x = [min(linear_regressor.X), max(linear_regressor.X)]
        line_y = [linear_function(linear_regressor.theta0,
                        linear_regressor.theta1, x) for x in line_x]
        plt.plot(line_x, line_y, 'b')
        plt.plot(linear_regressor.X, linear_regressor.Y, 'ro')
        t = threading.Thread(target=linear_regressor.train,
                kwargs={'max_iter':10e6, 'show':True, 'print_cost':args.cost})
        t.start()
        cid = plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
        plt.show()
    else:
        linear_regressor.train(print_cost=args.cost)

    if args.nb_iter:
        if args.cost:
            print("")
        print("The algorithm has converged after {} iterations".format(linear_regressor.nb_iter))
        if args.solution:
            print("")
    if args.solution:
        linear_regression_cost = get_cost(linear_regressor.theta0,
                linear_regressor.theta1, X, Y)
        real_theta0, real_theta1 = get_solution(X, Y)
        real_cost =  get_cost(real_theta0, real_theta1, X, Y)
        print_comparison(linear_regressor.theta0, linear_regressor.theta1,
            linear_regression_cost, real_theta0, real_theta1, real_cost)


if __name__ == '__main__':
    main()

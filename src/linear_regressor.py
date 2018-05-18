import numpy as np
import matplotlib.pyplot as plt
import time
import csv

from src.utils import get_cost, linear_function, sqrt

class LinearRegressor(object):
    def __init__(self, X, Y, theta0_init=0, theta1_init=0, learning_rate=1):
        self.X = X
        self.Y = Y
        self.theta0 = theta0_init
        self.theta1 = theta1_init
        self.learning_rate = learning_rate
        self.nb_iter = 0
        self.mu = 0
        self.sigma = 1
        self.line_x = [min(self.X), max(self.X)]

    def standardize(self):
        self.mu = np.mean([x for x in self.X])
        self.sigma = sqrt(sum([(x - self.mu)**2 for x in self.X]))
        self.X_S = [(x - self.mu) / self.sigma for x in self.X]

    def update_params(self):
        theta0 = self.learning_rate * sum([(self.theta1 * x + self.theta0) - y
                    for x, y in zip(self.X_S, self.Y)]) / len(self.X_S)
        theta1 = self.learning_rate * sum([((self.theta1 * x + self.theta0) - y) * x
                    for x, y in zip(self.X_S, self.Y)]) / len(self.X_S)
        self.theta0 -= theta0
        self.theta1 -= theta1

    def draw(self, sleep_duration=0.05):
        theta1_final = self.theta1 / self.sigma
        theta0_final = self.theta0 - (self.theta1 * self.mu) / self.sigma
        line_y = [linear_function(theta0_final, theta1_final, x) for x in self.line_x]
        plt.clf()
        plt.plot(self.X, self.Y, 'ro')
        plt.plot(self.line_x, line_y, 'b')
        plt.draw()
        time.sleep(sleep_duration)

    def train(self, epsilon=1e-20, max_iter=float("inf"), show=False,
                                                        print_cost=False):
        if show:
            time.sleep(1)
        self.standardize()
        last_cost = float("inf")
        while True:
            cost = get_cost(self.theta0, self.theta1, self.X_S, self.Y)
            if print_cost:
                print(cost)
            if abs(last_cost - cost) <= epsilon:
                break
            self.update_params()
            last_cost = cost
            max_iter -= 1
            if max_iter <= 0:
                 break
            if show and self.nb_iter % 200 == 0:
                self.draw()
            self.nb_iter += 1
        self.save(print_cost=print_cost)

    def save(self, print_cost=False):
        theta1_final = self.theta1 / self.sigma
        theta0_final = self.theta0 - (self.theta1 * self.mu) / self.sigma
        self.theta0 = theta0_final
        self.theta1 = theta1_final
        cost = get_cost(self.theta0, self.theta1, self.X, self.Y)
        if print_cost:
            print(cost, "\n")
        with open('params.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['theta0', 'theta1'])
            writer.writeheader()
            writer.writerow({'theta0': self.theta0, 'theta1':self.theta1})
            csvfile.close()

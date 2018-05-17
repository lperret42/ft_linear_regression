import numpy as np
import matplotlib.pyplot as plt

from linear_function import LinearFunction

class LinearRegressor(object):
    def __init__(self, X, Y, theta0_init=0, theta1_init=0, learning_rate=10e-3):
        self.X = X
        self.Y = Y
        self.theta0 = theta0_init
        self.theta1 = theta1_init
        self.learning_rate = learning_rate

        self.mu = 0
        self.sigma = 0

    def standardize(self):
        self.mu = np.mean([x for x in self.X])
        self.sigma = np.sqrt(sum([(x - self.mu)**2 for x in self.X]))
        self.X = [(x - self.mu) / self.sigma for x in self.X]

    def destandardize(self):
        self.X = [x * self.sigma + self.mu for x in self.X]

    def get_cost(self):
        return sum([(y - (self.theta1 * x + self.theta0)) ** 2
                            for x, y in zip(self.X, self.Y)]) /  (2* len(self.X))

    def update_params(self):
        theta0 = self.learning_rate * sum([(self.theta1 * x + self.theta0) - y
                                for x, y in zip(self.X, self.Y)]) / len(self.X)
        theta1 = self.learning_rate * sum([((self.theta1 * x + self.theta0) - y) * x
                                for x, y in zip(self.X, self.Y)]) / len(self.X)
        self.theta0 -= theta0
        self.theta1 -= theta1

    def train(self, epsilon=10e-3, max_iter=float("inf")):
        self.standardize()
        last_cost = float("inf")
        while True:
            cost = self.get_cost()
            if cost <= last_cost and (last_cost - cost) <= epsilon:
                break
            self.update_params()
            last_cost = cost
            max_iter -= 1
            if max_iter <= 0:
                 break

        theta1_final = self.theta1 / self.sigma
        theta0_final = self.theta0 - (self.theta1 * self.mu) / self.sigma
        self.theta0 = theta0_final
        self.theta1 = theta1_final
        self.destandardize()

    def show(self):
        linear_function = LinearFunction(self.theta0, self.theta1)
        line_x = [min(self.X), max(self.X)]
        line_y = [linear_function.evaluate(i) for i in line_x]
        plt.plot(line_x, line_y, 'b')
        plt.plot(self.X, self.Y, 'ro')
        plt.show()

class LinearFunction(object):
    def __init__(self, theta0, theta1):
        self.theta0 = theta0
        self.theta1 = theta1

    def evaluate(self, x):
        return self.theta0 + self.theta1 * x

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

class SimpleRegression:
    def __init__(self, dFrame, x, y, random_state=None):
        self.df = pd.read_csv(dFrame)
        self.x = x
        self.y = y
        self.clean = None
        if x not in self.df.columns or y not in self.df.columns:
            raise ValueError(f"{x}, {y} not in columns")

    def show_scatter(self):
        try:
            plt.scatter(self.clean[self.x], self.clean[self.y])
            plt.show()
        except:
            raise ValueError("No df!")

    def split(self, clear: bool = False):
        if self.x in self.df.columns and self.y in self.df.columns:
            self.clean = pd.DataFrame({
                f"{self.x}": self.df[self.x],
                f"{self.y}": self.df[self.y]
            }, index=self.df.index)
        else:
            raise ValueError(f"{self.x}, {self.y} not in columns")

        if clear:
            clean_mean = np.asarray(self.clean[self.x]).reshape(-1, 1).mean()
            for i, row in self.clean.iterrows():
                _p = (row[self.x] * 100 / clean_mean) - 100
                if _p < 0:
                    _p *= -1

                if _p > 60:
                    self.clean = self.clean.drop(i, axis="index")

        _80 = len(self.clean) * 0.8
        self.Train = self.clean.sample(n=int(_80), random_state=33)
        self.Xtrain = self.Train[self.x]
        self.Ytrain = self.Train[self.y]

        self.Test = self.clean.loc[~self.clean.index.isin(self.Train.index)]
        self.Xtest = self.Test[self.x]
        self.Ytest = self.Test[self.y]

    def fit(self, Test: bool = False):
        self._y = []

        x_mean = np.asarray(self.Xtrain).reshape(-1, 1).mean()
        y_mean = np.asarray(self.Ytrain).reshape(-1, 1).mean()

        numerator = 0
        denominator = 0

        for i, row in self.Train.iterrows():
            numerator += (row[self.x] - x_mean) * (row[self.y] - y_mean)
            denominator += (row[self.x] - x_mean) ** 2

        b = numerator / denominator
        a = y_mean - b * x_mean



        if Test:
            for i, row in self.Test.iterrows():
                self._y.append(a + b * row[self.x])

            plt.scatter(self.Xtest, self.Ytest)
            plt.plot(self.Xtest, self._y, color="red")
            plt.show()

        else:
            for i, row in self.Train.iterrows():
                self._y.append(a + b * row[self.x])

            plt.scatter(self.Xtrain, self.Ytrain)
            plt.plot(self.Xtrain, self._y, color="red")
            plt.show()

        return self._y

    def r_squared(self, Test: bool = False):
        if Test:
            return r2_score(np.asarray(self.Ytest).reshape(-1, 1), self._y)
        else:
            return r2_score(np.asarray(self.Ytrain).reshape(-1, 1), self._y)

    def mse(self, Test: bool = False):
        if Test:
            return mean_squared_error(np.asarray(self.Ytest).reshape(-1, 1), self._y)
        else:
            return mean_squared_error(np.asarray(self.Ytrain).reshape(-1, 1), self._y)


simple_EnigneSize = SimpleRegression('FuelConsumption.csv', "ENGINESIZE", "CO2EMISSIONS")

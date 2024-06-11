import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

class Knn:
    def __init__(self, k):
        self.k = k

    def euclidean_dis(self, x, sample):
        res = np.sqrt(np.sum(x - sample) ** 2)
        return res

    def fit(self, x_train, x_test, y_train, y_test):
        self.XTest = x_test
        self.XTrain = x_train
        self.YTest = y_test
        self.YTrain = y_train


        self.Predicts = []
        for i, x in enumerate(self.XTest):
            distance = np.array(self.dis(x))
            sort = self.YTrain[np.argsort(distance)[:self.k]]

            predict = Counter(sort).most_common(1)[0][0]
            self.Predicts.append(predict)


        return self.Predicts

    def dis(self, x):
        distance = []
        for n_x in self.XTrain:
            dis = self.euclidean_dis(n_x, x)
            distance.append(dis)

        return distance

    def accurancy(self):
        acc = 0
        for i in range(len(self.YTest)):
            if self.YTest[i] == self.Predicts[i]:
               acc += 1

        return (acc / len(self.YTest)) * 100


df = pd.read_csv("Social_Network_Ads.csv")
x = np.array(df[["Age", "EstimatedSalary"]])
y = np.array(df["Purchased"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

knn = Knn(13)
predict = knn.fit(x_train, x_test, y_train, y_test)
acc = knn.accurancy()
print(acc)

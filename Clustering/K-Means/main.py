import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class KMeans:
    def __init__(self, k_cluster, n_init, random_state):
        self.k_cluster = k_cluster
        self.n_init = n_init
        self.res = {}
        self.random_state = random_state

    def euclidean_dis(self, x, sample):
        res = np.sqrt(np.sum(x - sample) ** 2)
        return res

    def fit(self, x):
        cen_ = np.array(x.sample(n=self.k_cluster, random_state=self.random_state))
        x_array = np.array(x)

        for i in range(self.n_init):
            for k in range(self.k_cluster):
                self.res[f"Cluster{k + 1}"] = []

            for s in x_array:
                best = 0
                clus_ = 0
                for i_, c in enumerate(cen_):
                    if best == 0:
                        best = self.euclidean_dis(c, s)
                        clus_ = i_

                    if best >= self.euclidean_dis(c, s):
                        best = self.euclidean_dis(c, s)
                        clus_ = i_

                self.res[f"Cluster{clus_+1}"].append(s)

            cen_ = []
            for res in self.res:
                cen_.append(np.array(self.res[res]).mean(0))
                if i == self.n_init:
                    self.res[res] = np.array(self.res[res])


            cen_ = np.array(cen_)

        self.labels = np.array(cen_)

    def predict(self, x):
        x_array = np.array(x)

        predict = {}

        for k in range(self.k_cluster):
            predict[f"Cluster{k + 1}"] = []

        for s in x_array:
            best = 0
            clus_ = 0
            for i_, c in enumerate(self.labels):
                if best == 0:
                    best = self.euclidean_dis(c, s)
                    clus_ = i_

                if best >= self.euclidean_dis(c, s):
                    best = self.euclidean_dis(c, s)
                    clus_ = i_

            predict[f"Cluster{clus_ + 1}"].append(s)

        for pre in predict:
            predict[pre] = np.array(predict[pre])


        return predict


df = pd.read_csv("Cust_Segmentation.csv")
df = df.drop(["Address", "Customer Id"], axis=1)
df = df.dropna(axis=1, how="any")

x_train = df.iloc[:int((len(df)*0.8)), :]
x_test = df.iloc[:int((len(df)*0.2)), :]

Kmeans = KMeans(k_cluster=6, n_init=12, random_state=13)
Kmeans.fit(x_train)

k_labels = Kmeans.labels
y_predict = Kmeans.predict(x_test)


for i, k in enumerate(y_predict):
    cols = ["blue", "gray", "black", "skyblue", "pink", "gold"]
    plt.scatter(y_predict[k][:, 3], y_predict[k][:, 5], color=cols[i])

x_cen = k_labels[:, 3]
y_cen = k_labels[:, 5]

plt.scatter(x_cen, y_cen, color="red", marker="*")
plt.show()



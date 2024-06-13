import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs

class DBScan:
    def __init__(self, neighbors, radius):
        self.neighbors = neighbors
        self.radius = radius
        self.cluster = {}

    def euclidean_dis(self, x, sample):
        res = np.sqrt(np.sum(x - sample) ** 2)
        return res

    def core_points_(self, x):
        self.cores_ = []
        for i, r in x.iterrows():
            counter_n = 0
            for i_, r_ in x.iterrows():
                dis = self.euclidean_dis(np.array(r, dtype=float), np.array(r_, dtype=float))
                if dis != 0:
                    if dis <= self.radius:
                        counter_n += 1

                    if counter_n == self.neighbors:
                        self.cores_.append(i)
                        break

        return self.cores_

    def clustering(self, clus):
        def has_common_value(lst1, lst2):
            return any(val in lst2 for val in lst1)

        merged = True
        while merged:
            merged = False
            for i in range(len(clus)):
                for j in range(i + 1, len(clus)):
                    if has_common_value(clus[i], clus[j]):
                        clus[i] = list(set(clus[i] + clus[j]))
                        clus.pop(j)
                        merged = True
                        break
                if merged:
                    break
        return clus

    def clusters_(self, cores, x):
        cores_d = x[x.index.isin(cores)]
        clus_ = []

        for i, r in cores_d.iterrows():
            neighbors = [i]
            for i_, r_ in cores_d.iterrows():
                dis = self.euclidean_dis(np.array(r, dtype=float), np.array(r_, dtype=float))
                if dis != 0:
                    if dis <= self.radius:
                        neighbors.append(i_)

            clus_.append(neighbors)

        clus_ = self.clustering(clus_)

        for i, c in enumerate(clus_):
            self.cluster[f"Cluster{i+1}"] = c

        return self.cluster

    def join_(self, cores, x):
        not_c = x[~x.index.isin(cores)]
        self.out = []
        flg = 0
        for i, r in not_c.iterrows():
            for k in self.cluster:
                clus_ = x[x.index.isin(self.cluster[k])]
                for i_, r_ in clus_.iterrows():
                    dis = self.euclidean_dis(np.array(r, dtype=float), np.array(r_, dtype=float))
                    if dis <= self.radius:
                        self.cluster[k].append(i)
                        flg += 1
                        break

                if flg == 1:
                    flg = 0
                    break

            if flg == 0:
                self.out.append(i)

        return self.cluster

    def scatter_(self, x, y):
        for k, v in self.cluster.items():
            color = np.random.rand(3, )
            x_ = self.df[self.df.index.isin(v)]
            _x = np.array(x_.iloc[:, x], dtype=float)

            _y = np.array(x_.iloc[:, y], dtype=float)
            plt.scatter(_x, _y, color=color)

        out_d = self.df[self.df.index.isin(self.out)]
        x_o = np.array(out_d.iloc[:, x], dtype=float)
        y_o = np.array(out_d.iloc[:, y], dtype=float)
        plt.scatter(x_o, y_o, color="black")

        plt.show()

    def fit(self, x):
        cores = self.core_points_(x)
        self.df = x
        clusters = self.clusters_(cores, x)
        labels = self.join_(cores, x)



X1, y1 = make_blobs(n_samples=250, centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9, random_state=42)
plt.scatter(X1[:, 0], X1[:, 1], color="blue")

db = DBScan(neighbors=7, radius=0.7)
db.fit(pd.DataFrame(X1))
db.scatter_(0,1)
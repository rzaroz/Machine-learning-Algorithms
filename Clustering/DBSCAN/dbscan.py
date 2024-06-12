import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    def clusters_(self, cores, x):
        cores_d = x[x.index.isin(cores)]

        clus_ = []
        counter = 1
        for i, r in cores_d.iterrows():
            neighbors = []
            for i_, r_ in cores_d.iterrows():
                dis = self.euclidean_dis(np.array(r, dtype=float), np.array(r_, dtype=float))
                if dis != 0:
                    if dis <= self.radius:
                        neighbors.append(i_)

            if len(neighbors) == 0:
                neighbors = [i]
                clus_.append(neighbors)
            else:
                clus_.append(neighbors)


        for n in clus_:
            current = set(n)
            for n_ in clus_:
                plus = current.union(set(n_))
                if current != plus:
                    if len(plus) < len(n) + len(n_):
                        current = plus

            self.cluster[f"Cluster{counter}"] = list(current)
            counter += 1

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

        plt.show()

    def fit(self, x):
        cores = self.core_points_(x)
        self.df = x
        clusters = self.clusters_(cores, x)
        labels = self.join_(cores, x)



df = pd.read_csv("weather-stations20140101-20141231.csv")
df = df.drop(["BS%", "BS","Stn_Name", "Prov", "DwBS", "S_G", "D", "P%N", "Stn_No"], axis=1)
df = df.dropna(how='any', axis=0)
db = DBScan(neighbors=6, radius=10)
db.fit(df)
db.scatter_(14,13)
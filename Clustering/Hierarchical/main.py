import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Hierarchical:
    def __init__(self, n=2):
        self.n = n

    def euclidean_dis(self, x, sample):
        res = np.sqrt(np.sum(x - sample) ** 2)
        return res

    def distance_matrix(self, x):
        dis_mat = []
        x_arr = np.array(x, dtype=float)

        for r in x_arr:
            current_ = []
            for r_ in x_arr:
                current_.append(self.euclidean_dis(r, r_))

            dis_mat.append(current_)

        dis_mat = np.tril(np.array(dis_mat))
        dis_mat = pd.DataFrame(np.where(dis_mat == 0, np.inf, dis_mat))
        labels_ = [str(i) for i in range(len(dis_mat))]

        dis_mat["Cluster"] = labels_
        return dis_mat

    def average(self, x, y):
        res = (x + y) / 2
        return res

    def result(self, x):
        self.df_ = x.copy()
        dis_mat = self.distance_matrix(x)
        mat_arr = dis_mat.iloc[:, :-1]
        cluster_ = {}

        counter_ = 1
        while len(mat_arr) > self.n:
            min_c = mat_arr.min().idxmin()
            min_r = mat_arr[min_c].idxmin()

            avg = self.average(np.array(self.df_.loc[min_r], dtype=float), np.array(self.df_.loc[min_c], dtype=float))
            self.df_.loc[self.df_.index.max() + 1] = avg

            if dis_mat.loc[min_r]["Cluster"] in cluster_.keys() and dis_mat.loc[min_c]["Cluster"] in cluster_.keys():
                cluster_[f"Cluster{counter_}"] = cluster_[dis_mat.loc[min_r]["Cluster"]] + cluster_[dis_mat.loc[min_c]["Cluster"]]
                cluster_.pop(dis_mat.loc[min_r]["Cluster"])
                cluster_.pop(dis_mat.loc[min_c]["Cluster"])
                dis_mat = self.update_(dis_mat, [min_r, min_c], avg, f"Cluster{counter_}")
                counter_ += 1

            elif not dis_mat.loc[min_r]["Cluster"] in cluster_.keys() and dis_mat.loc[min_c]["Cluster"] in cluster_.keys():
                cluster_[dis_mat.loc[min_c]["Cluster"]].append(min_r)
                dis_mat = self.update_(dis_mat, [min_r, min_c], avg, dis_mat.loc[min_c]["Cluster"])

            elif dis_mat.loc[min_r]["Cluster"] in cluster_.keys() and not dis_mat.loc[min_c]["Cluster"] in cluster_.keys():
                cluster_[dis_mat.loc[min_r]["Cluster"]].append(min_c)
                dis_mat = self.update_(dis_mat, [min_r, min_c], avg, dis_mat.loc[min_r]["Cluster"])
            else:
                cluster_[f"Cluster{counter_}"] = [min_r, min_c]
                dis_mat = self.update_(dis_mat, [min_r, min_c], avg, f"Cluster{counter_}")
                counter_ += 1

            mat_arr = dis_mat.iloc[:, :-1]
        return cluster_

    def update_(self, dis_mat, min_coords, avg, name):
        new_c = dis_mat.index.max() + 1
        mat_arr = dis_mat.iloc[:, :-1]
        val_ = pd.DataFrame({new_c: [np.inf for i in range(len(mat_arr))]}, index=mat_arr.index)

        dis_mat = pd.concat([mat_arr, val_, dis_mat["Cluster"]], axis=1)

        new_d = [self.euclidean_dis(avg, np.array(r, dtype=float)) for i_, r in self.df_.iterrows()]
        new_d[-1] = np.inf
        new_d.append(name)

        dis_mat.loc[dis_mat.index.max() + 1] = new_d

        dis_mat = dis_mat.drop(min_coords, axis=0).drop(min_coords, axis=1)
        self.df_ = self.df_.drop(min_coords, axis=0)

        return dis_mat

    def labels_(self, res: dict):
        i_ = 1
        self.DataFrame["Label"] = np.zeros(len(self.DataFrame), dtype=int)
        for k, v in res.items():
            for i in v:
                self.DataFrame.loc[i, "Label"] = i_
            i_ += 1

        self.labels = np.array(self.DataFrame["Label"])

        return self.labels

    def scatter_(self):
        for k, v in self.res.items():
            color = np.random.rand(3, )
            x_ = self.DataFrame[self.DataFrame.index.isin(v)]
            _x = np.array(x_.iloc[:, 3:4], dtype=float)

            _y = np.array(x_.iloc[:, 4:5], dtype=float)
            plt.scatter(_x, _y, color=color)

        plt.show()

    def fit(self, x):
        self.DataFrame = x
        result = self.result(x)
        self.res = result
        self.labels_(result)


df = pd.read_csv("Cust_Segmentation.csv")
df = df.drop(["Address", "Customer Id"], axis=1)
df = df.dropna(axis=1, how="any")

Hierarchical_ = Hierarchical(5)
Hierarchical_.fit(df)
Hierarchical_.scatter_()
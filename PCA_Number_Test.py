import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
    with open("D:/cer_electricity/F+C_Set/F+C_Set.csv", 'r') as file:
        f_reader = csv.reader(file)
        x = []
        for row in f_reader:
            x.append(row[1:69])

        # x = np.array(x)
        x_scaler = StandardScaler()
        x = x_scaler.fit_transform(x)

        # pca = PCA()
        pca = PCA(n_components=0.90)  # 0.90 -> keep 19 features, 0.95 -> 27, 0.99 -> 44
        pca.fit(x)
        ratio = pca.explained_variance_ratio_
        print("pca.components_", pca.components_.shape)
        print("pca_var_ratio", pca.explained_variance_ratio_.shape)
        # 绘制图形
        plt.plot([i for i in range(x.shape[1])],
                 [np.sum(ratio[:i + 1]) for i in range(x.shape[1])])
        plt.xticks(np.arange(x.shape[1], step=5))
        plt.yticks(np.arange(0, 1.01, 0.05))
        plt.grid()
        plt.show()


if __name__ == '__main__':
    main()

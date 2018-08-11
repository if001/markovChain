import numpy as np
from sklearn.cluster import KMeans
import pickle


class MyKmeans():
    def __init__(self, num_of_cluster,  save_file, data_array=[], init_flag="init"):
        self.num_of_cluster = num_of_cluster
        data_array = np.array(data_array[::])
        if init_flag == "init":
            self.__k_means = KMeans(
                n_clusters=num_of_cluster,
                n_init=10,               # 異なるセントロイドの初期値を用いたk-meansあるゴリmズムの実行回数
                max_iter=300,            # k-meansアルゴリズムの内部の最大イテレーション回数
                tol=1e-04,               # 収束と判定するための相対的な許容誤差
                random_state=0          # セントロイドの初期化に用いる乱数発生器の状態
            ).fit(data_array)
            print("k-means init and save")
            self.save_model(save_file)
        else:
            self.__k_means = self.load_model(save_file)
        self.__predict_array = self.__k_means.predict(data_array)

    def get_sse(self):
        return self.__k_means.inertia_

    def get_predict(self):
        return self.__predict_array[::]

    def get_cluster(self, vec):
        return self.__k_means.predict(vec)[0]

    def save_model(self, save_file):
        print("save model ", save_file)
        pickle.dump(self.__k_means, open(save_file, 'wb'))

    def load_model(self, save_file):
        print("load model ", save_file)
        return pickle.load(open(save_file, 'rb'))

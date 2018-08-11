from file_operator import FileOperator
from string_operator import StringOperator
from probability_state import ProbabilityState
from probability_state_kvs import ProbabilityStateKvs
from w2v import Word2Vec
from common.const import Const
from k_means import MyKmeans
import networkx as nx


import sys
import numpy.random as np_rand
import numpy as np
from graphviz import Digraph
from graphviz import Graph

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.metrics import silhouette_samples


def load_data():
    read = FileOperator.f_open(Const.UNIQ_SRC_FILE)
    unique_char_set = read[-1].split(",")
    w2v = Word2Vec(Const.W2V_SRC_FILE, Const.W2V_WEIGHT_FILE,
                   Const.WORD_FEAT_LEN, "load")

    data_array = []
    for word in unique_char_set:
        data_array.append(w2v.str_to_vector(word))
    kmeans = MyKmeans(Const.NUM_OF_CLUSTER,
                      Const.KMEANS_SAVE_FILE, data_array, "load")
    return unique_char_set, w2v, kmeans


def draw_graph():
    unique_char_set, _, kmeans = load_data()
    print("number of unique word:", len(unique_char_set))
    cluster_array = kmeans.get_predict()
    print(cluster_array)

    belong_word = [[] for i in range(Const.NUM_OF_CLUSTER)]
    for i in range(len(cluster_array)):
        belong_word[cluster_array[i]].append(unique_char_set[i])
    for i in range(Const.NUM_OF_CLUSTER):
        print("cluster: ", i, ":", belong_word[i])
    exit(0)
    prob_state = ProbabilityState(Const.NUM_OF_CLUSTER, Const.PROB_FILE,"load")
    tra_cnt = prob_state.get_trainsition_cnt()
    g = Digraph(format="png")
    g.attr("node", shape="circle")
    for i in range(len(tra_cnt)):
        for j in range(len(tra_cnt)):
            sys.stdout.write("\r progress: %d / %d" % (i, j))
            sys.stdout.flush()
            
            g.edge(str(i), str(j),
                   label='{:.2f}'.format(tra_cnt[i][j] / max(tra_cnt[i])),
                   penwidth=str(tra_cnt[i][j] / max(tra_cnt[i])))
    g.render(Const.GRAPH_IMG)


def draw_graph_nx():
    unique_char_set, _, kmeans = load_data()
    print("number of unique word:", len(unique_char_set))
    cluster_array = kmeans.get_predict()
    print(cluster_array)

    belong_word = [[] for i in range(Const.NUM_OF_CLUSTER)]
    for i in range(len(cluster_array)):
        belong_word[cluster_array[i]].append(unique_char_set[i])
    for i in range(Const.NUM_OF_CLUSTER):
        print("cluster: ", i, ":", belong_word[i])

    prob_state = ProbabilityState(Const.NUM_OF_CLUSTER, Const.PROB_FILE,"load")
    tra_cnt = prob_state.get_trainsition_cnt()
    G = nx.DiGraph()
    # G = nx.Digraph(format="png")
    G.add_nodes_from(range(0, Const.NUM_OF_CLUSTER))

    for i in range(len(tra_cnt)):
        for j in range(len(tra_cnt)):
            sys.stdout.write("\r progress: %d / %d" % (i, len(tra_cnt)*len(tra_cnt)))
            sys.stdout.flush()
            G.add_edge(i, j)

            # g.edge(str(i), str(j),
            #        label='{:.2f}'.format(tra_cnt[i][j] / max(tra_cnt[i])),
            #        penwidth=str(tra_cnt[i][j] / max(tra_cnt[i])))
    nx.draw_networkx(G)
    plt.show()


            
def call_sse():
    read = FileOperator.f_open(Const.UNIQ_SRC_FILE)
    w2v = Word2Vec(Const.W2V_SRC_FILE, Const.W2V_WEIGHT_FILE,
                   Const.WORD_FEAT_LEN, "load")

    unique_char_set = read[-1].split(",")
    print("number of unique word:", len(unique_char_set))
    data_array = []
    for word in unique_char_set:
        data_array.append(w2v.str_to_vector(word))

    sse_list = []
    num_of_cluster_list = range(100, 2000, 100)
    for num_of_cluster in num_of_cluster_list:
        print(num_of_cluster)
        kmeans = MyKmeans(
            num_of_cluster, Const.KMEANS_SAVE_FILE, data_array, "init")
        print(kmeans.get_sse())
        sse_list.append(kmeans.get_sse())

    plt.plot(num_of_cluster_list, sse_list, marker='o')
    # plt.show()
    plt.savefig(Const.SSE_IMG)


def silhouette():
    unique_char_set, w2v, kmeans = load_data()
    print("number of unique word:", len(unique_char_set))
    data_array = []
    for word in unique_char_set:
        data_array.append(w2v.str_to_vector(word))
    kmeans = MyKmeans(Const.NUM_OF_CLUSTER,
                      Const.KMEANS_SAVE_FILE, data_array, "init")
    predict_array = kmeans.get_predict()
    cluster_labels = np.unique(predict_array)       # y_kmの要素の中で重複を無くす
    n_clusters = cluster_labels.shape[0]     # 配列の長さを返す

    # シルエット係数を計算
    silhouette_vals = silhouette_samples(
        data_array, predict_array, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []

    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[predict_array == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)               # 色の値をセット
        plt.barh(range(y_ax_lower, y_ax_upper),            # 水平の棒グラフのを描画（底辺の範囲を指定）
                 c_silhouette_vals,               # 棒の幅
                 height=1.0,                      # 棒の高さ
                 edgecolor='none',                # 棒の端の色
                 color=color)                     # 棒の色
        yticks.append((y_ax_lower + y_ax_upper) /
                      2)          # クラスタラベルの保油次位置を追加
        y_ax_lower += len(c_silhouette_vals)              # 底辺の値に棒の幅を追加

    silhouette_avg = np.mean(silhouette_vals)                 # シルエット係数の平均値
    plt.axvline(silhouette_avg, color="red", linestyle="--")    # 係数の平均値に破線を引く
    plt.yticks(yticks, cluster_labels + 1)                     # クラスタレベルを表示
    plt.ylabel('Cluster')
    plt.xlabel('silhouette cofficient')
    plt.show()


def main():
    arg = sys.argv[-1]
    if arg == "-g":
        draw_graph()
    elif arg == "-n":
        draw_graph_nx()
    elif arg == "-e":  # エルボー法
        call_sse()
    elif arg == "-s":  # シルエット法
        silhouette()
    else:
        print("invalid argument")
        print("-g : 遷移図")
        print("-n : 遷移図 nx")
        print("-e : エルボー法")
        print("-s : シルエット法")


if __name__ == "__main__":
    main()

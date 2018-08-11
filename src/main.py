from file_operator import FileOperator
from string_operator import StringOperator
from probability_state import ProbabilityState
from probability_state_kvs import ProbabilityStateKvs
from w2v import Word2Vec
from common.const import Const
import sys
from k_means import MyKmeans
import numpy.random as np_rand
import numpy as np


def init_data():
    print("src file: ", Const.SRC_FILE)
    sentence_list = FileOperator.f_open(Const.SRC_FILE)
    sentence_list = StringOperator.split_sentence(sentence_list)

    flatten_word_list = StringOperator.array_string_to_flatten(
        sentence_list)
    unique_char_set = StringOperator.array_char_to_unique(flatten_word_list)
    print("unique char set len :", len(unique_char_set))
    FileOperator.f_write(Const.UNIQ_SRC_FILE, unique_char_set)
    print("save unique file: ", Const.UNIQ_SRC_FILE)
    prob_state = ProbabilityState(Const.NUM_OF_CLUSTER, Const.PROB_FILE, "init")
    prob_state.save_prob(Const.PROB_FILE)
    w2v = Word2Vec(Const.W2V_SRC_FILE, Const.W2V_WEIGHT_FILE, Const.WORD_FEAT_LEN, "init")

    data_array = []
    for word in unique_char_set:
        data_array.append(w2v.str_to_vector(word))
    MyKmeans(Const.NUM_OF_CLUSTER, Const.KMEANS_SAVE_FILE, data_array, "init")


def load_data():
    read = FileOperator.f_open(Const.UNIQ_SRC_FILE)
    unique_char_set = read[-1].split(",")
    prob_state = ProbabilityState(Const.NUM_OF_CLUSTER, Const.PROB_FILE, "load")
    w2v = Word2Vec(Const.W2V_SRC_FILE, Const.W2V_WEIGHT_FILE, Const.WORD_FEAT_LEN, "load")
    data_array = []
    for word in unique_char_set:
        data_array.append(w2v.str_to_vector(word))
    kmeans = MyKmeans(Const.NUM_OF_CLUSTER, Const.KMEANS_SAVE_FILE, data_array, "load")
    return prob_state, w2v, kmeans, unique_char_set


def learn_word():
    print("src file: ", Const.SRC_FILE)
    sentence_list = FileOperator.f_open(Const.SRC_FILE)
    sentence_list = StringOperator.split_sentence(sentence_list)
    prob_state, w2v, kmeans, _ = load_data()

    cnt = 0
    for sentence in sentence_list:
        sys.stdout.write("\r progress: %d / %d" % (cnt, len(sentence_list)))
        sys.stdout.flush()
        for i in range(len(sentence) - 2):
            vec = w2v.str_to_vector(sentence[i]).reshape(1, -1)
            cluster = kmeans.get_cluster(vec)
            next_vec = w2v.str_to_vector(sentence[i + 1]).reshape(1, -1)
            next_cluster = kmeans.get_cluster(next_vec)
            prob_state.count_up_trainsition(cluster, next_cluster)
        cnt += 1
    prob_state.save_prob(Const.PROB_FILE)
    print()
    print("end")


def predict_word():
    prob_state, w2v, kmeans, unique_char_set = load_data()

    for _ in range(10):
        sentence = ""
        word = StringOperator.choice_char(unique_char_set)[-1]
        loop = True
        cluster_move = []
        while(loop):
            sentence += word
            vec = w2v.str_to_vector(word).reshape(1, -1)
            cluster = kmeans.get_cluster(vec)
            cluster_move.append(cluster)
            next_cluster = prob_state.get_next_word_base_prob(cluster)
            arr = kmeans.get_predict()
            cluster_idx_arr = []
            for i in range(len(arr)):
                if arr[i] == next_cluster:
                    cluster_idx_arr.append(i)
            idx = np_rand.choice(cluster_idx_arr)
            word = unique_char_set[idx]
            if word == '。':
                sentence += '。'
                loop = False
            if len(sentence) >= 100:
                loop = False
        print(sentence)
        print(cluster_move)
        print("--------------")


def main():
    arg = sys.argv[-1]
    if arg == "--init":
        init_data()
    elif arg == "-l":
        learn_word()
    elif arg == "-p":
        predict_word()
    else:
        print("invalid argument")
        print("-l : learn")
        print("-p : predict")


if __name__ == "__main__":
    main()

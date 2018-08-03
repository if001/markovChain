import numpy as np
import numpy.random as np_rand


class ProbabilityState():
    def __init__(self, unique_char_set):
        """
        p(y | x) = prob[x][y]
        sim_cnt[x][y]
        """
        self.__unique_char_set = unique_char_set
        tmp_arr = [
            [0 for i in range(len(unique_char_set))]
            for j in range(len(unique_char_set))
        ]
        self.__transition_cnt = np.array(tmp_arr)

    def cal_cond_prob(self, char, transition_char):
        c_idx = self.__unique_char_set.index(char)
        t_c_idx = self.__unique_char_set.index(transition_char)
        return self.__transition_cnt[c_idx][t_c_idx] / \
            sum(self.__transition_cnt[c_idx])

    def get_given_char_prob_dist(self, char):
        c_idx = self.__unique_char_set.index(char)
        return self.__transition_cnt[c_idx][::]

    def count_up_trainsition(self, char, transition_char):
        char_idx = self.__unique_char_set.index(char)
        trainsition_char_idx = self.__unique_char_set.index(transition_char)
        self.__transition_cnt[char_idx][trainsition_char_idx] += 1

    def get_trainsition_cnt(self):
        return self.__transition_cnt[::]

    def get_next_word_base_prob(self, char):
        __p = self.get_given_char_prob_dist(char)
        return np_rand.choice(self.__unique_char_set, 1, p=__p)[0]

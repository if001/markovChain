import numpy as np
import numpy.random as np_rand


class ProbabilityState():
    def __init__(self, state_len, fname, init_flag="init"):
        """
        p(y | x) = prob[x][y]
        sim_cnt[x][y]
        """
        self.__state_len = state_len
        tmp_arr = [
            [0 for i in range(state_len)]
            for j in range(state_len)
        ]
        if init_flag == "init":
            print("init trainsition count")
            self.__transition_cnt = np.array(tmp_arr)
        else:
            print("load trainsition count:", fname)
            self.__transition_cnt = self.load_prob(fname)

    def cal_cond_prob(self, state, transition_state):
        return self.__transition_cnt[state][transition_state] / \
            sum(self.__transition_cnt[state])

    def get_given_state_prob_dist(self, state):
        return self.__transition_cnt[state] / sum(self.__transition_cnt[state])

    def count_up_trainsition(self, state, transition_state):
        self.__transition_cnt[state][transition_state] += 1

    def get_trainsition_cnt(self):
        return self.__transition_cnt[::]

    def get_next_word_base_prob(self, state):
        __p = self.get_given_state_prob_dist(state)
        return np_rand.choice(self.__state_len, 1, p=__p)[0]

    def save_prob(self, fname):
        print("save trainsition count:", fname)
        np.save(fname, self.__transition_cnt)

    def load_prob(self, fname):
        return np.load(fname)

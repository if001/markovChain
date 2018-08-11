import sys
import numpy as np
import numpy.random as np_rand
from prob_kvs import ProbKvs


class ProbabilityStateKvs():
    def __init__(self, unique_word_set, fname, init_flag="init"):
        """
        p(y | x) = prob[x][y]
        """
        self.__unique_word_set = unique_word_set
        print(len(self.__unique_word_set))
        print(np.square(len(self.__unique_word_set)))
        exit(0)
        self.prob_kvs = ProbKvs(fname)
        if init_flag == "init":  # kvs init
            cnt = 0
            for word_set1 in self.__unique_word_set:
                for word_set2 in self.__unique_word_set:
                    sys.stdout.write("\r progress: %d / %d" %
                                     (cnt, len(self.__unique_word_set) * len(self.__unique_word_set)))
                    sys.stdout.flush()
                    self.prob_kvs.put(word_set1 + word_set2, [0])
                    cnt += 1

    def get_given_char_prob_dist(self, char):
        array_cnt = []
        for unique_word in self.__unique_word_set:
            array_cnt.append(self.prob_kvs.get(char + unique_word[-1]))
        return array_cnt / sum(array_cnt)

    def count_up_trainsition(self, char, transition_char):
        print(self.prob_kvs.get(char + transition_char))
        cnt = self.prob_kvs.get(char + transition_char)[-1]
        self.prob_kvs.put(char + transition_char, [int(cnt) + 1])

    def get_next_word_base_prob(self, char):
        __p = self.get_given_char_prob_dist(char)
        return np_rand.choice(self.__unique_word_set, 1, p=__p)[0]

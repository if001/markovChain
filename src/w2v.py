# 単語をベクトル化

import gensim
import numpy as np


class Word2Vec():
    def __init__(self,
                 train_src_file,
                 word2vec_wight_file,
                 word_feat_len=25,
                 init_flag="init"):
        sentences = gensim.models.word2vec.Text8Corpus(train_src_file)
        if init_flag == "init":
            print("train " + train_src_file)
            print("save " + word2vec_wight_file)
            self.model = gensim.models.word2vec.Word2Vec(
                sentences, size=word_feat_len, window=5, workers=4, min_count=1, hs=1)
            self.model.save(word2vec_wight_file)
        else:
            print("load " + word2vec_wight_file)
            self.model = gensim.models.word2vec.Word2Vec.load(
                word2vec_wight_file)

    def get_some_word(self, vec, num):
        return self.model.most_similar([vec], [], num)

    def vec_to_word(self, vec):
        return self.model.most_similar([vec], [], 1)[0][0]

    def vec_to_some_word(self, vec, num):
        return self.model.most_similar([vec], [], num)

    def str_to_vector(self, st):
        return self.model.wv[st]


def main():
    pass
    # net.load_model()
    # vec = net.get_vector("博士")
    # vec = net.get_vector("明智")
    # vec = Word2Vec.get_vector("怪盗")
    # plot(vec)

    # vec = np.array(vec,dtype='f')
    # word = net.get_word(vec)
    # print("word",word)

    # net.get_word()


if __name__ == "__main__":
    main()

import projectRoot
from itertools import chain
import numpy as np


class StringOperator():
    @classmethod
    def array_string_to_flatten(cls, array_string):
        return list(chain.from_iterable(array_string))

    @classmethod
    def array_char_to_unique(cls, array_char):
        return list(set(array_char))

    @classmethod
    def choice_char(cls, char_list):
        return np.random.choice(char_list, 1)

    @classmethod
    def split_sentence(cls, array_sentence):
        """
        無理やり整形、もっと綺麗にかける
        """
        arr = []
        for sentence in array_sentence:
            splited = sentence.split('\n')
            while('' in splited):
                splited.remove('')
            splited = sentence.split(' ')
            if ('。\n' in splited):
                splited.remove('。\n')
            if ('…。\n' in splited):
                splited.remove('…。\n')
            if ("——。\n" in splited):
                splited.remove("——。\n")
            splited.append('。')
            # splited.append('\n')
            while('' in splited):
                splited.remove('')
            arr.append(splited)
        return arr[::]

import projectRoot
from itertools import chain


class StringOperator():
    @classmethod
    def array_string_to_flatten(cls, array_string):
        return list(chain.from_iterable(array_string))

    @classmethod
    def array_char_to_unique(cls, array_char):
        return list(set(array_char))

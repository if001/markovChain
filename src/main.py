from file_operator import FileOperator
from string_operator import StringOperator
from probability_state import ProbabilityState


def main():
    sentence_list = FileOperator.f_open("./test2.txt")

    flatten_char = StringOperator.array_string_to_flatten(sentence_list)

    unique_char_set = StringOperator.array_char_to_unique(flatten_char)
    # FileOperator.f_write("./test2.txt", unique_char_set)
    prob_state = ProbabilityState(unique_char_set)
    print(unique_char_set)
    for sentence in sentence_list:
        for i in range(len(sentence) - 1):
            print(sentence[i], sentence[i + 1])
            prob_state.count_up_trainsition(sentence[i], sentence[i + 1])

    t = prob_state.get_trainsition_cnt()
    w = prob_state.get_next_word_base_prob("あ")


if __name__ == "__main__":
    main()

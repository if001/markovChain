class Const():
    DIR = "./date/"
    FILE_PREFIX = "files_all_rnp_split20000"

    SRC_FILE = DIR + FILE_PREFIX + ".txt"
    W2V_SRC_FILE = DIR + "files_all_rnp.txt"
    UNIQ_SRC_FILE = DIR + "unique_" + FILE_PREFIX + ".txt"

    PROB_FILE = DIR + "prob_" + FILE_PREFIX + ".npy"
    W2V_WEIGHT_FILE = DIR + "text8.model"
    KMEANS_SAVE_FILE = DIR + "k_means_" + FILE_PREFIX + ".sav"

    WORD_FEAT_LEN = 25
    NUM_OF_CLUSTER = 2000

    IMG_DIR = "./image/"
    SSE_IMG = IMG_DIR + "sse.png"
    GRAPH_IMG = IMG_DIR + "graph"

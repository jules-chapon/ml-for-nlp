"""Constants"""

###############################################################
#                                                             #
#                             PATHS                           #
#                                                             #
###############################################################

OUTPUT_FOLDER = "output"

REMOTE_TRAINING_FOLDER = "remote_training"


###############################################################
#                                                             #
#                          DATASETS                           #
#                                                             #
###############################################################

### LOCAL

LOCAL_TRAIN_DATASET_1_PATH = "data/input/train_1.csv"

LOCAL_TEST_DATASET_1_PATH = "data/input/test_1.csv"

STOPWORDS_PATH = "data/inputs/stopwords.txt"

### HUGGING FACE

HF_FULL_DATASET_1_NAME = "Ateeqq/AI-and-Human-Generated-Text"


###############################################################
#                                                             #
#                      REMOTE TRAINING                        #
#                                                             #
###############################################################

GIT_USER = "jules-chapon"  # make sure it is the good name of your GitHub account

GIT_REPO = "ml-for-nlp"  # Make sure it is the good name of your repository

NOTEBOOK_ID = "ml-for-nlp"

KAGGLE_DATASET_LIST = []


###############################################################
#                                                             #
#                        FIXED VALUES                         #
#                                                             #
###############################################################

RANDOM_SEED = 42

VALID_RATIO = 0.1

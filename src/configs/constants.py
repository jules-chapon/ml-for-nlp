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

### DOWNLOAD DATASET

REPO_URL = "https://github.com/sakibsh/LLM"
TARGET_FOLDER = "data"
TEMPORARY_FOLDER = "__tmp"
DATA_GITHUB_FOLDER = "data/input/github"

### FILENAMES

BARD_ESSAY = "BARD_essay.csv"
BARD_POETRY = "BARD_poetry.csv"
BARD_CODE = "BARD_pycode.csv"
BARD_STORY = "BARD_story.csv"

GPT_ESSAY = "ChatGPT_essay.csv"
GPT_POETRY = "ChatGPT_poetry.csv"
GPT_CODE = "ChatGPT_pycode.csv"
GPT_STORY = "ChatGPT_story.csv"

HUMAN_ESSAY_1 = "human_essay_1.csv"
HUMAN_ESSAY_2 = "human_essay_2.csv"
HUMAN_ESSAY_3 = "human_essay_hewlett.csv"
HUMAN_ESSAY_4 = "human_essay_hugg.csv"
HUMAN_POETRY = "human_poetry.csv"
HUMAN_CODE = "human_code.csv"
HUMAN_STORY = "human_stories.csv"


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

GPT_LABEL = 1

BARD_LABEL = 2

HUMAN_LABEL = 0

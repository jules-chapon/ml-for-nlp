"""Parameters for ML models"""

from src.configs import constants, names


###############################################################
#                                                             #
#                     EXPERIMENTS CONFIGS                     #
#                                                             #
###############################################################

# ID_EXPERIMENT: ID_EMBEDDING, ID_CLASSIFIER, ID_PARAMS
# Example : 012

# names.EMBEDDING_TYPE = [names.TF_IDF]
# IF names.EMBEDDING_TYPE = names.TF_IDF (1)
# PARAMS = {MAX_FEATURES : int}
# IF names.EMBEDDING_TYPE = names.PUNCTUATION (2)
# PARAMS = {}

# names.CLASSIFIER_TYPE = [names.LGBM, names.RANDOM_FOREST, names.NAIVE_BAYES]
# IF names.CLASSIFIER_TYPE = names.LGBM (1)
# PARAMS = {BOOSTING_TYPE : str, N_ESTIMATORS : int, MAX_DEPTH : int, NUM_LEAVES : int, SUBSAMPLE : float, N_JOBS : int, LEARNING_RATE : float, VERBOSE: int, RANDOM_STATE : int}
# IF names.CLASSIFIER_TYPE = names.RANDOM_FOREST (2)
# PARAMS = {N_ESTIMATORS : int, MAX_DEPTH : int, MIN_SAMPLES_SPLIT : int, MIN_SAMPLES_LEAF : int, BOOTSRAP: bool, N_JOBS : int, RANDOM_STATE : int}
# IF names.CLASSIFIER_TYPE = names.NAIVE_BAYES (3)
# PARAMS = {ALPHA : float}


EXPERIMENTS_CONFIGS = {
    211: {
        names.DEVICE: names.CPU,
        ### EMBEDDING : TF-IDF
        names.EMBEDDING_TYPE: names.PUNCTUATION,
        # PARAMS
        names.EMBEDDING_PARAMS: {},
        ### CLASSIFIER : LGBM
        names.CLASSIFIER_TYPE: names.LGBM,
        # PARAMS
        names.CLASSIFIER_PARAMS: {
            names.BOOSTING_TYPE: "gbdt",
            names.N_ESTIMATORS: 100,
            names.MAX_DEPTH: 10,
            names.NUM_LEAVES: 31,
            names.SUBSAMPLE: 0.8,
            names.N_JOBS: -1,
            names.LEARNING_RATE: 0.05,
            names.VERBOSE: -1,
            names.RANDOM_STATE: constants.RANDOM_SEED,
        },
    },
    111: {
        names.DEVICE: names.CPU,
        ### EMBEDDING : TF-IDF
        names.EMBEDDING_TYPE: names.TF_IDF,
        # PARAMS
        names.EMBEDDING_PARAMS: {
            names.MAX_FEATURES: 10000,
        },
        ### CLASSIFIER : LGBM
        names.CLASSIFIER_TYPE: names.LGBM,
        # PARAMS
        names.CLASSIFIER_PARAMS: {
            names.BOOSTING_TYPE: "gbdt",
            names.N_ESTIMATORS: 100,
            names.MAX_DEPTH: 10,
            names.NUM_LEAVES: 31,
            names.SUBSAMPLE: 0.8,
            names.N_JOBS: -1,
            names.LEARNING_RATE: 0.05,
            names.VERBOSE: -1,
            names.RANDOM_STATE: constants.RANDOM_SEED,
        },
    },
    121: {
        names.DEVICE: names.CPU,
        ### EMBEDDING : TF-IDF
        names.EMBEDDING_TYPE: names.TF_IDF,
        # PARAMS
        names.EMBEDDING_PARAMS: {
            names.MAX_FEATURES: 10000,
        },
        ### CLASSIFIER : RANDOM FOREST
        names.CLASSIFIER_TYPE: names.RANDOM_FOREST,
        # PARAMS
        names.CLASSIFIER_PARAMS: {
            names.N_ESTIMATORS: 100,
            names.MAX_DEPTH: 10,
            names.MIN_SAMPLES_SPLIT: 2,
            names.MIN_SAMPLES_LEAF: 1,
            names.BOOTSTRAP: True,
            names.N_JOBS: -1,
            names.RANDOM_STATE: constants.RANDOM_SEED,
        },
    },
    131: {
        names.DEVICE: names.CPU,
        ### EMBEDDING : TF-IDF
        names.EMBEDDING_TYPE: names.TF_IDF,
        # PARAMS
        names.EMBEDDING_PARAMS: {
            names.MAX_FEATURES: 10000,
        },
        ### CLASSIFIER : NAIVE BAYES
        names.CLASSIFIER_TYPE: names.NAIVE_BAYES,
        # PARAMS
        names.CLASSIFIER_PARAMS: {names.ALPHA: 1.0},
    },
    # Add more experiments as needed
}

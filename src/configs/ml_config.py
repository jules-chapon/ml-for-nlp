"""Parameters for ML models"""

from src.configs import names


###############################################################
#                                                             #
#                     EXPERIMENTS CONFIGS                     #
#                                                             #
###############################################################

# names.EMBEDDING_TYPE = [names.TF_IDF]
# IF names.EMBEDDING_TYPE = names.TF_IDF
# PARAMS = {MAX_FEATURES : int, }


EXPERIMENTS_CONFIGS = {
    0: {
        # EMBEDDING
        names.EMBEDDING_TYPE: names.TF_IDF,
        # PARAMS
        names.MAX_FEATURES: 10000,
    }
    # Add more experiments as needed
}

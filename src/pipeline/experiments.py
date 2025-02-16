"""Functions to define different experiments"""

from src.pipeline.pipeline import Pipeline


def init_pipeline_from_config(id_experiment: int, iteration: int) -> Pipeline | None:
    """
    Initialize a pipeline for a given experiment.

    Args:
        id_experiment (int): ID of the experiment.
        iteration (int): Iteration number.

    Returns:
        Pipeline | None: Pipeline with the parameters of the given experiment.
    """
    return Pipeline(id_experiment=id_experiment, iteration=iteration)

from dataclasses import dataclass

@dataclass
class Parameters():
    """Class for reserving the hyperparameters."""
    # todo: consider to set hyperparam according to the graph characteristics (num of nodes etc.)
    WINDOW: int = 6
    MIN_COUNT: int = 1
    BATCH_WORDS: int = 4

    DIMENSIONS: int = 128
    WALK_LENGTH: int = 30
    NUM_WALKS: int = 200
    SEED: int = 42

    # number of target node pairs in the directory
    NUM_SAMPLES: int = 1000

    # directory
    # DIR : str = "/tmp/sc_graph_samples"
from dataclasses import dataclass

@dataclass
class Parameters():
    """Class for reserving the hyperparameters."""
    # todo: consider to set hyperparam according to the graph characteristics (num of nodes etc.)
    PATTERN: str = "avg"
    WINDOW: int = 10
    MIN_COUNT: int = 1
    BATCH_WORDS: int = 5

    DIMENSIONS: int = 128
    WALK_LENGTH: int = 20
    NUM_WALKS: int = 20
    SEED: int = 42

    # number of target node pairs in the directory
    NUM_SAMPLES_PER_LABEL: int = 500

    # attri2vec
    BATCH_SIZE: int = 50
    EPOCHS: int = 5
    LEARNING_RATE: float = 1e-3
    WORKERS: int = 1

    # GCN
    EPOCHS_GCN: int = 50
    LEARNING_RATE_GCN: float = 0.01
from dataclasses import dataclass


@dataclass
class Parameters():
    """
    Class for reserving all the hyperparameters. Values
    can be used by initialized an object of this Class
    and call its respective attibutes
    """
    # pattern of processing embedding vectors of
    # two target nodes. E.g., "avg": (v1+v2)/2,
    # please refer to ../preprocessor/graph_processing.py
    # function "operate" for more details
    PATTERN_GRAPH: str = "avg"
    PATTERN_TEXT: str = "avg"

    # parameters for node2vec
    WINDOW: int = 10
    MIN_COUNT: int = 1
    BATCH_WORDS: int = 5

    WALK_LENGTH: int = 20
    NUM_WALKS: int = 20
    SEED: int = 42

    # dimension of embedding vectors
    DIMENSIONS: int = 128

    # number of data samples to be used to generate graphs
    # E.g., setting this parameter as 100 means
    # (theoretically) generating 100 neg and 100 pos samples
    NUM_SAMPLES_PER_LABEL: int = 7000

    # realistic testset
    POS_NEG_RATIO: float = 0.1

    # bert embedding
    MAX_LENGTH: int = 256

    # attri2vec
    BATCH_SIZE: int = 50
    EPOCHS: int = 5
    LEARNING_RATE: float = 1e-3
    WORKERS: int = 1

    # General params for DL
    DROPOUT: float = 0.5
    ACTIVATION: str = "relu"
    ES_PATIENCE: int = 5

    # GCN
    EPOCHS_GCN: int = 50
    LEARNING_RATE_GCN: float = 0.01

    # RGCN
    NUM_BASES: int = 20
    EPOCHS_RGCN: int = 50
    LEARNING_RATE_RGCN: float = 0.01

    # GAT
    EPOCHS_GAT: int = 50
    LEARNING_RATE_GAT: float = 0.01

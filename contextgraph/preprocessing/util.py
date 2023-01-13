""" Utility functions for preprocessing
"""

import os
from contextgraph import config as cg_config


def ensure_graph_data_dir():
    """ Ensure the directory where the preprocessed data
        will be placed exists.
    """

    if not os.path.exists(cg_config.graph_data_dir):
        os.mkdir(cg_config.graph_data_dir)

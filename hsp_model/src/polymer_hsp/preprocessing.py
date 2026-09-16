"""
Data preprocessing utilities
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

def remove_empty_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove descriptor columns that contain only zeros and replace missing values.
    :param df: input dataframe
    :return: cleaned dataframe
    """

    cleaned = df.copy()

    cleaned = cleaned.fillna(0)

    initial_features = cleaned.shape[1]
    cleaned = cleaned.loc[:, (cleaned != 0).any(axis=0)]

    removed = initial_features - cleaned.shape[1]
    logger.info(f"Removed {removed} empty descriptor columns")

    return cleaned

"""
Input/output for the polymer HSP modelling pipeline
"""

from pathlib import Path
import logging

import pandas as pd

logger = logging.getLogger(__name__)

def read_dataset(filepath: str | Path, sheet_name: str) -> pd.DataFrame:
    """
    Load Hansen Solubility Parameter dataset from an Excel workbook
    :param filepath: excel workbook filepath
    :param sheet_name: Worksheet containing the target property (e.g. "DeltaD", "DeltaP", "DeltaH").
    :return: Loaded dataset
    """

    filepath = Path(filepath)


    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    logger.info(f"Loading sheet {sheet_name} from {filepath}")

    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    except ValueError as exc:
        raise ValueError(f"Excel Sheet {sheet_name} not found in {filepath.name}") from exc

    # if present remove index column
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    logger.info(f"Loaded dataset with shape =  {df.shape}")

    return df
import numpy as np
import pandas as pd  
from pathlib import Path

from numpy.lib._stride_tricks_impl import sliding_window_view


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load and preprocess data from a CSV file. Remove rows with unlabeled data.

    Args:
        data_path (Path): Path to the CSV data file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with unlabeled data removed.
    Raises:
        FileNotFoundError: If the specified data file does not exist.
        ValueError: If the data is empty after removing unlabeled data and dropping NaN values.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file {data_path} does not exist.")
    df = pd.read_csv(data_path)
    df = remove_unlabeled_data(df)
    df = df.dropna()
    return df


def remove_unlabeled_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with unlabeled data (where labels == -1).

    Args:
        data (pd.DataFrame): Input DataFrame containing a 'labels' column.

    Returns:
        pd.DataFrame: DataFrame with unlabeled data removed.
    """
    return data[data["labels"]!=-1]


def convert_to_np(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert DataFrame to numpy arrays, separating labels, experiment IDs, and features.

    Args:
        data (pd.DataFrame): Input DataFrame containing 'labels', 'exp_ids', and feature columns.

    Returns:
        tuple: A tuple containing:
            - labels (np.ndarray): Array of labels
            - exp_ids (np.ndarray): Array of experiment IDs
            - data (np.ndarray): Combined array of current and voltage features
    """
    # Label und experiment IDs in NumPy-Array abtrennen
    labels = data["labels"].to_numpy()
    exp_ids = data["exp_ids"].to_numpy()

    # Current und Volatge per sorted (korrekte Reihenfolge) filtern und sortieren
    current_cols = sorted(col for col in data.columns if col.startswith("I"))
    voltage_cols = sorted(col for col in data.columns if col.startswith("V"))

    # gefilterte und sortierte current und voltage colums in NumPy-Array umwandeln
    current = data[current_cols].to_numpy()
    voltage = data[voltage_cols].to_numpy()

    # current und voltage zu 3D-Array stapeln
    data = np.stack((current, voltage), axis=-1)


    return labels, exp_ids, data


def create_sliding_windows_first_dim(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Create sliding windows over the first dimension of a 3D array.
    
    Args:
        data (np.ndarray): Input array of shape (n_samples, timesteps, features)
        sequence_length (int): Length of each window
    
    Returns:
        np.ndarray: Windowed data of shape (n_windows, sequence_length*timesteps, features)
    """
    data = data
    sequence_length = sequence_length
    n_samples, timesteps, features = data.shape
    if sequence_length > n_samples:
        raise ValueError(
            f"sequence_length ({sequence_length}) must be smaller than n_samples ({n_samples})."
        )
    windows = sliding_window_view(data, window_shape=sequence_length, axis=0)
    n_windows = windows.shape[0]
    windows = windows.reshape((n_windows, sequence_length*timesteps, features))

    return windows

def get_welding_data(path: Path, n_samples: int | None = None, return_sequences: bool = False, sequence_length: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load welding data from CSV or cached numpy files.

    If numpy cache files don't exist, loads from CSV and creates cache files.
    If cache files exist, loads directly from them.

    Args:
        path (Path): Path to the CSV data file.
        n_samples (int | None): Number of samples to sample from the data. If None, all data is returned.
        return_sequences (bool): If True, return sequences of length sequence_length.
        sequence_length (int): Length of sequences to return.
    Returns:
        tuple: A tuple containing:
            - np.ndarray: Array of welding data features
            - np.ndarray: Array of labels
            - np.ndarray: Array of experiment IDs
    """
    pass

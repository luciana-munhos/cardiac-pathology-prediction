import numpy as np
from sklearn.model_selection import train_test_split
from features import process_dataset, process_data_best_features
from sklearn.preprocessing import MinMaxScaler

def load_training_data(metadata_path, base_path, id_range, use_completed_seg=False, random_state=42, dropFeatures=False):
    """
    Loads and preprocesses the full training dataset.

    Parameters:
        metadata_path: Path to the metadata CSV file
        base_path: Base path to training data
        id_range: Range of training sample IDs
        use_completed_seg: Whether to use completed segmentations
        random_state: Seed for reproducibility
        dropFeatures: Whether to drop features (only done in one case)

    Returns:
        X_train: Feature matrix
        y_train: Corresponding labels
    """

    # Load metadata
    metadata = process_dataset(
        metadata_path=metadata_path,
        base_path=base_path,
        id_range=id_range,
        use_completed_seg=use_completed_seg,
        dropFeatures=dropFeatures
    )

    # Shuffle the data
    np.random.seed(random_state)
    N = len(metadata)
    order = np.arange(0, N)
    np.random.shuffle(order)
    metadata = metadata[order, :]

    # Separate labels and features
    y_train = metadata[:, 0]
    X_train = metadata[:, 1:]

    return X_train, y_train

def load_training_data_14_features(metadata_path, base_path, id_range, use_completed_seg=False, random_state=42):
    """
    Loads and preprocesses the full training dataset, considering only the 14 best features in order of feature importance.

    Parameters:
        metadata_path: Path to the metadata CSV file
        base_path: Base path to training data
        id_range: Range of training sample IDs
        use_completed_seg: Whether to use completed segmentations
        random_state: Seed for reproducibility

    Returns:
        X_train: Feature matrix
        y_train: Corresponding labels
    """

    # Load metadata
    metadata = process_data_best_features(
        metadata_path=metadata_path,
        base_path=base_path,
        id_range=id_range,
        use_completed_seg=use_completed_seg
    )

    # Shuffle the data
    np.random.seed(random_state)
    N = len(metadata)
    order = np.arange(0, N)
    np.random.shuffle(order)
    metadata = metadata[order, :]

    # Separate labels and features
    y_train = metadata[:, 0]
    X_train = metadata[:, 1:]

    return X_train, y_train

# to use a subset of the training set to train
def load_and_split_data_train(metadata_path, base_path, id_range, use_completed_seg=False, test_size=0.2, random_state=42):
    # Load metadata
    metadata = process_dataset(
        metadata_path=metadata_path,
        base_path=base_path,
        id_range=id_range,
        use_completed_seg=use_completed_seg
    )

    # Shuffle the data
    N = len(metadata)
    order = np.arange(0,N)
    np.random.shuffle(order)
    metadata = metadata[order, :]

    # Separate labels and features
    y = metadata[:, 0]
    X = metadata[:, 1:]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    return X_train, X_test, y_train, y_test

def normalize_features(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
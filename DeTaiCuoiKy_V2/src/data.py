import pandas as pd
from .config import DATA_PROCESSED

def load_data():
    X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").squeeze("columns")
    labels = pd.read_csv(DATA_PROCESSED / "label_names.csv").squeeze("columns").tolist()
    return X_train, X_test, y_train, y_test, labels
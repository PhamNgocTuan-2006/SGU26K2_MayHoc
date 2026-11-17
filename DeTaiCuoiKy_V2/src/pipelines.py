from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from .config import RANDOM_STATE

def scaled(clf):
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

def get_baselines():
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logreg": scaled(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "tree_depth3": DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
    }
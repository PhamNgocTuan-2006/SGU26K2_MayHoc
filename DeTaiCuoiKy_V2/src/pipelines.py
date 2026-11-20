from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from .config import RANDOM_STATE

def scaled(clf):
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

def get_baselines():
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logreg": scaled(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "tree_depth3": DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
    }

def get_classic():
    return {
        "knn_k5": scaled(KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
    }

def get_advanced():
    return {
        "mlp_64": scaled(MLPClassifier(
            hidden_layer_sizes=(64,), alpha=1e-4, early_stopping=True,
            n_iter_no_change=10, max_iter=200, random_state=RANDOM_STATE)),
        "hgb_default": HistGradientBoostingClassifier(
            max_iter=100, early_stopping=False, random_state=RANDOM_STATE),
    }

def get_xgb():      # tuy chon, chi dung neu da cai xgboost
    from xgboost import XGBClassifier
    return {"xgb_default": XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1, tree_method="hist",
        objective="multi:softprob", n_jobs=-1, random_state=RANDOM_STATE)}
def get_final():
    # mo hinh cuoi da khoa, chon theo quy tac trong README
    return HistGradientBoostingClassifier(
        class_weight="balanced", early_stopping=False, random_state=RANDOM_STATE
    )
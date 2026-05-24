import os

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC

from utils.file_io import load_jsonl

DEFAULT_C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
DEFAULT_MAX_ITER_VALUES = [1000]
DEFAULT_CV_FOLDS = 5


def _build_probe(
    probe_type: str, c_values: list, max_iter_values: list, n_train: int = 500
):
    if probe_type == "logistic":
        return (
            LogisticRegression(solver="lbfgs", n_jobs=-1),
            {"C": c_values, "max_iter": max_iter_values},
            True,
        )
    if probe_type == "ridge":
        return (
            RidgeClassifier(),
            {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
            True,
        )
    if probe_type == "svm_linear":
        return (
            CalibratedClassifierCV(LinearSVC(dual="auto", max_iter=2000), cv=3),
            {"estimator__C": c_values, "estimator__max_iter": max_iter_values},
            True,
        )
    if probe_type == "svm_rbf":
        return (
            SVC(kernel="rbf"),
            {
                "C": c_values,
                "gamma": ["scale", "auto", 1e-4, 1e-3],
                "max_iter": [2000],
            },
            True,
        )
    if probe_type == "mlp":
        return (
            MLPClassifier(random_state=42, early_stopping=True),
            {
                "hidden_layer_sizes": [(256,), (512,), (256, 128)],
                "alpha": [1e-4, 1e-3, 1e-2],
                "max_iter": [500],
            },
            True,
        )
    if probe_type == "knn":
        k_max = min(21, n_train)
        k_values = [k for k in [3, 5, 7, 11, 15, 21] if k < k_max]
        if not k_values:
            k_values = [1]
        return (
            KNeighborsClassifier(n_jobs=-1),
            {
                "n_neighbors": k_values,
                "metric": ["cosine", "euclidean"],
                "weights": ["uniform", "distance"],
            },
            True,
        )
    if probe_type == "nearest_centroid":
        return (NearestCentroid(), {}, False)
    if probe_type == "lda":
        return (
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            {},
            False,
        )
    if probe_type == "random_forest":
        return (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "n_estimators": [100, 300],
                "max_depth": [None, 16, 32],
                "min_samples_leaf": [1, 3],
            },
            True,
        )
    raise ValueError(f"Unknown probe type: {probe_type}")


def _get_clf_params(clf, probe_type: str) -> dict:
    if probe_type == "nearest_centroid":
        return {"metric": clf.metric, "shrink_threshold": clf.shrink_threshold}
    if probe_type == "lda":
        return {"solver": clf.solver, "shrinkage": str(clf.shrinkage)}
    return {}


def _layer_sort_key(k: str):
    return (0, int(k)) if k.lstrip("-").isdigit() else (1, k)


def compute_layer_stability(layer_f1_scores: list[float]) -> dict:
    arr = np.array(layer_f1_scores)
    if arr.size == 0:
        return {"mean_f1": 0.0, "std_f1": 0.0, "max_f1": 0.0, "best_layer": -1}
    return {
        "mean_f1": float(np.mean(arr)),
        "std_f1": float(np.std(arr)),
        "max_f1": float(np.max(arr)),
        "best_layer": int(np.argmax(arr)),
    }


def _build_selectivity_clf(probe_type: str, seed: int = 42):
    builders = {
        "logistic": lambda: LogisticRegression(
            solver="lbfgs", max_iter=1000, C=1.0, n_jobs=-1
        ),
        "ridge": lambda: RidgeClassifier(alpha=1.0),
        "svm_linear": lambda: LinearSVC(max_iter=2000, dual="auto"),
        "svm_rbf": lambda: SVC(kernel="rbf", max_iter=2000),
        "mlp": lambda: MLPClassifier(
            hidden_layer_sizes=(256,), max_iter=1000, random_state=seed
        ),
        "knn": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "nearest_centroid": lambda: NearestCentroid(),
        "lda": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1
        ),
    }
    return builders[probe_type]()


def train_selectivity_control(
    X_train, y_train, X_test, y_test, probe_type="logistic", seed=42
):
    rng = np.random.RandomState(seed)
    y_shuffled = rng.permutation(y_train)
    clf = _build_selectivity_clf(probe_type, seed)
    clf.fit(X_train, y_shuffled)
    shuffled_preds = clf.predict(X_test)
    shuffled_acc = float(accuracy_score(y_test, shuffled_preds))
    shuffled_f1 = float(
        f1_score(y_test, shuffled_preds, average="macro", zero_division=0)
    )
    return shuffled_acc, shuffled_f1


def build_error_detection_labels(
    model_stem: str,
    dataset_name: str,
    approach: str,
    reasoning: str,
    options_mode: str,
    options_order: str,
    n_shots: int,
    split_data,
):
    config_name = f"{approach}_{reasoning}_{options_mode}"
    if options_order != "default":
        config_name += f"_{options_order}"
    if n_shots > 0:
        config_name += f"_{n_shots}shots"
    pred_path = f"./data/sample_generations/{model_stem}/{dataset_name}/prompting/{config_name}.jsonl"
    if not os.path.exists(pred_path):
        return None
    preds = load_jsonl(pred_path)
    idx_to_correct = {r["idx"]: r.get("correct", 0) for r in preds}
    labels = []
    for sample in split_data.samples:
        labels.append(idx_to_correct.get(sample.idx, 0))
    return np.array(labels)

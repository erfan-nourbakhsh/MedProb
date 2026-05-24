import argparse
import os
import sys
from datetime import datetime

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logistic_regression import (
    DEFAULT_CV_FOLDS,
    DEFAULT_C_VALUES,
    DEFAULT_MAX_ITER_VALUES,
    _build_probe,
    _get_clf_params,
    _layer_sort_key,
    build_error_detection_labels,
    compute_layer_stability,
    train_selectivity_control,
)
from utils.constants import (
    ALL_MODELS,
    APPROACHES,
    OPTION_MODES,
    OPTION_ORDER_MODES,
    PROBE_TASKS,
    PROBE_TYPES,
    REASONING_STRATEGIES,
)
from utils.file_io import append_jsonl, load_npz_with_integrity_check
from utils.loaders import load_dataset

def _build_config_name(
    approach: str,
    reasoning: str,
    options_mode: str,
    options_order: str,
    n_shots: int,
) -> str:
    config_name = f"{approach}_{reasoning}_{options_mode}"
    if options_order != "default":
        config_name += f"_{options_order}"
    if n_shots > 0:
        config_name += f"_{n_shots}shots"
    return config_name

def parse_args():
    p = argparse.ArgumentParser(
        description="Probe with default-order train embeddings and reordered test embeddings"
    )
    p.add_argument("--dataset", type=str, default="PATH-VQA")
    p.add_argument("--dataset_dir", type=str, default="./samples/PATH-VQA")
    p.add_argument("--model_stem", type=str, required=True, choices=ALL_MODELS)
    p.add_argument("--approach", type=str, required=True, choices=APPROACHES)
    p.add_argument("--reasoning", type=str, required=True, choices=REASONING_STRATEGIES)
    p.add_argument("--options_mode", type=str, required=True, choices=OPTION_MODES)
    p.add_argument(
        "--train_options_order",
        type=str,
        default="default",
        choices=OPTION_ORDER_MODES,
        help="Option order used for train embeddings and train labels",
    )
    p.add_argument(
        "--test_options_order",
        type=str,
        required=True,
        choices=OPTION_ORDER_MODES,
        help="Option order used for test embeddings and test labels",
    )
    p.add_argument(
        "--probe_task", type=str, default="answer_decoding", choices=PROBE_TASKS
    )
    p.add_argument("--probe_type", type=str, default="logistic", choices=PROBE_TYPES)
    p.add_argument("--normalize", type=int, default=1, choices=[0, 1])
    p.add_argument("--cv", type=int, default=DEFAULT_CV_FOLDS)
    p.add_argument("--selectivity", type=int, default=1, choices=[0, 1])
    p.add_argument("--c_values", type=str, default=None)
    p.add_argument("--max_iter_values", type=str, default=None)
    p.add_argument("--n_shots", type=int, default=0)
    p.add_argument("--extraction_position", type=str, default="last_input_token")
    return p.parse_args()

def main():
    args = parse_args()
    c_values = (
        [float(x) for x in args.c_values.split(",")]
        if args.c_values
        else DEFAULT_C_VALUES
    )
    max_iter_values = (
        [int(x) for x in args.max_iter_values.split(",")]
        if args.max_iter_values
        else DEFAULT_MAX_ITER_VALUES
    )
    print("=" * 70)
    print(
        f"CROSS-CONDITION PROBING — {args.probe_task.upper()} / {args.probe_type.upper()}"
    )
    print("=" * 70)
    print(f"Dataset:            {args.dataset}")
    print(f"Model:              {args.model_stem}")
    print(f"Approach:           {args.approach}")
    print(f"Reasoning:          {args.reasoning}")
    print(f"Options Mode:       {args.options_mode}")
    print(f"Train Order:        {args.train_options_order}")
    print(f"Test Order:         {args.test_options_order}")
    print(f"N Shots:            {args.n_shots}")
    print(f"Probe Task:         {args.probe_task}")
    print(f"Probe Type:         {args.probe_type}")
    print(f"Position:           {args.extraction_position}")
    print(f"Selectivity:        {bool(args.selectivity)}")
    print(f"CV folds:           {args.cv}")
    print("=" * 70)
    train_data, test_data = load_dataset(args.dataset_dir, args.dataset)
    if args.probe_task == "answer_decoding":
        all_labels = sorted(
            set(s.answer_label for s in train_data.samples + test_data.samples)
        )
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        y_train = np.array([label_to_idx[s.answer_label] for s in train_data.samples])
        y_test = np.array([label_to_idx[s.answer_label] for s in test_data.samples])
        idx_to_label = {v: k for k, v in label_to_idx.items()}
    elif args.probe_task == "error_detection":
        y_train = build_error_detection_labels(
            args.model_stem,
            args.dataset,
            args.approach,
            args.reasoning,
            args.options_mode,
            args.train_options_order,
            args.n_shots,
            train_data,
        )
        y_test = build_error_detection_labels(
            args.model_stem,
            args.dataset,
            args.approach,
            args.reasoning,
            args.options_mode,
            args.test_options_order,
            args.n_shots,
            test_data,
        )
        if y_train is None or y_test is None:
            print(
                "[ERROR] Prompting predictions not found for one or both train/test orders"
            )
            sys.exit(1)
        label_to_idx = {0: 0, 1: 1}
        idx_to_label = {0: "incorrect", 1: "correct"}
    else:
        raise ValueError(f"Unknown probe task: {args.probe_task}")
    train_config = _build_config_name(
        args.approach,
        args.reasoning,
        args.options_mode,
        args.train_options_order,
        args.n_shots,
    )
    test_config = _build_config_name(
        args.approach,
        args.reasoning,
        args.options_mode,
        args.test_options_order,
        args.n_shots,
    )
    position_suffix = (
        ""
        if args.extraction_position == "last_input_token"
        else f"_{args.extraction_position}"
    )
    train_embed_path = f"./data/sample_features/{args.model_stem}/{args.dataset}/{train_config}{position_suffix}"
    test_embed_path = f"./data/sample_features/{args.model_stem}/{args.dataset}/{test_config}{position_suffix}"
    train_npz_path = os.path.join(train_embed_path, "train_embeddings.npz")
    test_npz_path = os.path.join(test_embed_path, "test_embeddings.npz")
    if not os.path.exists(train_npz_path):
        print(f"[ERROR] Train embeddings not found: {train_npz_path}")
        sys.exit(1)
    if not os.path.exists(test_npz_path):
        print(f"[ERROR] Test embeddings not found: {test_npz_path}")
        sys.exit(1)
    train_npz = load_npz_with_integrity_check(train_npz_path)
    test_npz = load_npz_with_integrity_check(test_npz_path)
    train_keys = list(train_npz.files)
    test_keys = list(test_npz.files)
    assert set(train_keys) == set(test_keys), "train/test NPZ layer keys differ"
    layer_keys = sorted(train_keys, key=_layer_sort_key)
    print(f"Layers: {len(layer_keys)}")
    task_dir = f"{args.probe_task}_{args.probe_type}_default_train"
    transfer_name = f"train_{train_config}__test_{test_config}"
    out_root = (
        f"./data/sample_generations/{args.model_stem}/{args.dataset}/"
        f"{task_dir}/{transfer_name}{position_suffix}"
    )
    os.makedirs(out_root, exist_ok=True)
    best_params_dir = os.path.join(out_root, "best_params")
    os.makedirs(best_params_dir, exist_ok=True)
    all_layer_f1 = []
    for layer_key in tqdm(
        layer_keys, desc=f"Cross-condition probing per layer ({args.probe_type})"
    ):
        train_embeddings = train_npz[layer_key]
        test_embeddings = test_npz[layer_key]
        if args.normalize:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            train_embeddings = scaler.fit_transform(train_embeddings)
            test_embeddings = scaler.transform(test_embeddings)
        assert train_embeddings.shape[0] == len(y_train), "train size mismatch"
        assert test_embeddings.shape[0] == len(y_test), "test size mismatch"
        n_unique = len(np.unique(y_train))
        n_cv = min(args.cv, n_unique)
        base_clf, param_grid, use_grid = _build_probe(
            args.probe_type,
            c_values,
            max_iter_values,
            n_train=train_embeddings.shape[0],
        )
        if use_grid:
            from sklearn.model_selection import GridSearchCV

            grid_search = GridSearchCV(
                base_clf,
                param_grid,
                cv=n_cv,
                scoring="f1_macro",
                n_jobs=-1,
                verbose=0,
                refit=True,
            )
            grid_search.fit(train_embeddings, y_train)
            best_params = grid_search.best_params_
            best_cv_score = float(grid_search.best_score_)
            best_clf = grid_search.best_estimator_
        else:
            from sklearn.model_selection import cross_val_score

            base_clf.fit(train_embeddings, y_train)
            best_params = _get_clf_params(base_clf, args.probe_type)
            cv_scores = cross_val_score(
                _build_probe(
                    args.probe_type,
                    c_values,
                    max_iter_values,
                    n_train=train_embeddings.shape[0],
                )[0],
                train_embeddings,
                y_train,
                cv=n_cv,
                scoring="f1_macro",
                n_jobs=-1,
            )
            best_cv_score = float(np.mean(cv_scores))
            best_clf = base_clf
        from sklearn.metrics import accuracy_score, f1_score

        predictions = best_clf.predict(test_embeddings)
        y_pred = [int(p) for p in predictions]
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        all_layer_f1.append(f1)
        sel_acc, sel_f1, selectivity = None, None, None
        if args.selectivity:
            sel_acc, sel_f1 = train_selectivity_control(
                train_embeddings,
                y_train,
                test_embeddings,
                y_test,
                probe_type=args.probe_type,
            )
            selectivity = acc - sel_acc
        print(
            f"\nLayer {layer_key} | best_params={best_params} | "
            f"cv_f1={best_cv_score:.4f} | test_acc={acc:.4f} test_f1={f1:.4f}"
            + (f" | selectivity={selectivity:.4f}" if selectivity is not None else "")
        )
        pred_path = os.path.join(out_root, f"layer{layer_key}.jsonl")
        with open(pred_path, "w", encoding="utf-8"):
            pass
        for sample, pred in zip(test_data.samples, predictions):
            record = {
                "idx": sample.idx,
                "gt": (
                    label_to_idx.get(sample.answer_label, int(y_test[sample.idx]))
                    if args.probe_task == "answer_decoding"
                    else int(y_test[sample.idx])
                ),
                "pred": int(pred),
                "gt_label": (
                    sample.answer_label
                    if args.probe_task == "answer_decoding"
                    else str(int(y_test[sample.idx]))
                ),
                "pred_label": idx_to_label.get(int(pred), "?"),
                "layer": layer_key,
                "probe_task": args.probe_task,
                "probe_type": args.probe_type,
                "train_options_order": args.train_options_order,
                "test_options_order": args.test_options_order,
            }
            append_jsonl(pred_path, record)
        params_path = os.path.join(best_params_dir, f"layer{layer_key}.jsonl")
        with open(params_path, "w", encoding="utf-8"):
            pass
        params_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": args.dataset,
            "model_stem": args.model_stem,
            "approach": args.approach,
            "reasoning": args.reasoning,
            "options_mode": args.options_mode,
            "train_options_order": args.train_options_order,
            "test_options_order": args.test_options_order,
            "train_config": train_config,
            "test_config": test_config,
            "probe_task": args.probe_task,
            "probe_type": args.probe_type,
            "extraction_position": args.extraction_position,
            "layer": layer_key,
            "cv_folds": args.cv,
            "best_params": {k: str(v) for k, v in best_params.items()},
            "best_cv_f1_macro": best_cv_score,
            "test_accuracy": acc,
            "test_f1_macro": f1,
            "pred_path": pred_path,
        }
        if sel_acc is not None:
            params_record["shuffled_accuracy"] = sel_acc
            params_record["shuffled_f1"] = sel_f1
            params_record["selectivity"] = selectivity
        append_jsonl(params_path, params_record)
    stability = compute_layer_stability(all_layer_f1)
    print("\n" + "=" * 70)
    print("LAYER-WISE STABILITY")
    print(f"  mean_f1:    {stability['mean_f1']:.4f}")
    print(f"  std_f1:     {stability['std_f1']:.4f}")
    print(f"  max_f1:     {stability['max_f1']:.4f}")
    print(f"  best_layer: {stability['best_layer']}")
    stability_path = os.path.join(out_root, "layer_stability.jsonl")
    with open(stability_path, "w", encoding="utf-8"):
        pass
    append_jsonl(
        stability_path,
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": args.dataset,
            "model_stem": args.model_stem,
            "train_config": train_config,
            "test_config": test_config,
            "train_options_order": args.train_options_order,
            "test_options_order": args.test_options_order,
            "probe_task": args.probe_task,
            "probe_type": args.probe_type,
            "extraction_position": args.extraction_position,
            **stability,
        },
    )
    print(f"\nDone. Predictions: {out_root}")
    print(f"Best params: {best_params_dir}")

if __name__ == "__main__":
    main()

import os
import time
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ============================================================
# 可配置参数区域
# ============================================================
TARGET_COL = "SeriousDlqin2yrs"
RANDOM_STATE = 42

DT_PARAMS = {
    "criterion": "gini",       # 或 "entropy"
    "max_depth": 4,
    "min_samples_split": 30,
    "min_samples_leaf": 200,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",  # 处理类别不平衡
}

# 阈值搜索范围：在验证集上遍历以下候选值，选取 F1 最高的阈值
# 若不需要阈值优化，将 THRESHOLD_SEARCH 设为 False，则默认使用 0.5
THRESHOLD_SEARCH = True
THRESHOLD_CANDIDATES = np.arange(0.1, 0.6, 0.01)  # 0.10, 0.11, ..., 0.59
# ============================================================


def evaluate_binary(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def find_best_threshold(y_true, y_prob, candidates):
    """在候选阈值中选取验证集 F1 最高的阈值。"""
    best_threshold, best_f1 = 0.5, -1.0
    for t in candidates:
        y_pred_t = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred_t, zero_division=0)
        if score > best_f1:
            best_f1, best_threshold = score, t
    return float(best_threshold), float(best_f1)


def drop_unnamed_columns(df):
    """移除历史 index 导出产生的 Unnamed 列，避免特征列错位。"""
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


def load_fold_data(train_path, val_path, target_col):
    """读取单折数据并做列一致性检查，返回 (X_train, y_train, X_val, y_val, val_df)。"""
    train_df = drop_unnamed_columns(pd.read_csv(train_path))
    val_df = drop_unnamed_columns(pd.read_csv(val_path))

    if target_col not in train_df.columns or target_col not in val_df.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在，请检查数据文件列名。")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int)
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col].astype(int)

    if list(X_train.columns) != list(X_val.columns):
        missing = [c for c in X_train.columns if c not in X_val.columns]
        extra = [c for c in X_val.columns if c not in X_train.columns]
        if missing or extra:
            raise ValueError(f"特征列不一致。缺失: {missing}，多余: {extra}")
        X_val = X_val[X_train.columns]

    return X_train, y_train, X_val, y_val, val_df


def print_vertical_metrics(title, metrics):
    print(f"\n===== {title} =====")
    label_width = max(len(str(k)) for k in metrics.keys())
    for key, value in metrics.items():
        print(f"{str(key):<{label_width}} : {value}")


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    train_dir = os.path.join(project_root, "data", "processed", "five_folds_standardized")
    val_dir = os.path.join(project_root, "data", "processed", "five_folds_standardized")

    fold_metrics = []
    oof_predictions = []

    for fold in range(1, 6):
        train_path = os.path.join(train_dir, f"fold_{fold}_train_scaled.csv")
        val_path = os.path.join(val_dir, f"fold_{fold}_val_scaled.csv")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"找不到训练集文件: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"找不到验证集文件: {val_path}")

        X_train, y_train, X_val, y_val, val_df = load_fold_data(
            train_path, val_path, TARGET_COL
        )

        # 训练
        model = DecisionTreeClassifier(**DT_PARAMS)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        # 预测概率
        y_prob = model.predict_proba(X_val)[:, 1]

        # 阈值选取
        if THRESHOLD_SEARCH:
            threshold, _ = find_best_threshold(y_val, y_prob, THRESHOLD_CANDIDATES)
        else:
            threshold = 0.5
        y_pred = (y_prob >= threshold).astype(int)

        metrics = evaluate_binary(y_val, y_pred, y_prob)
        metrics["fold"] = fold
        metrics["threshold"] = round(threshold, 4)
        metrics["train_time_seconds"] = round(train_time, 4)
        fold_metrics.append(metrics)

        oof_predictions.append(pd.DataFrame({
            "fold": fold,
            "sample_index": val_df.index,
            "y_true": y_val.to_numpy(),
            "y_pred": y_pred,
            "y_prob": y_prob,
        }))

    # 汇总
    metrics_df = pd.DataFrame(fold_metrics)
    oof_df = pd.concat(oof_predictions, ignore_index=True)

    # OOF 整体指标使用各折阈值预测的 y_pred（已存入 oof_df）
    overall_metrics = evaluate_binary(oof_df["y_true"], oof_df["y_pred"], oof_df["y_prob"])

    print("\n===== 5-Fold Metrics Table =====")
    print(metrics_df[
        ["fold", "threshold", "accuracy", "precision", "recall", "f1", "auc", "train_time_seconds"]
    ].to_string(index=False))

    print_vertical_metrics("Overall OOF Metrics IN Decision Tree", overall_metrics)


if __name__ == "__main__":
    main()

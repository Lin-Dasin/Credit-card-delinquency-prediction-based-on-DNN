import argparse
import os
import time
import warnings

import pandas as pd
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluate_binary(y_true, y_pred, y_prob):
	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"f1": float(f1_score(y_true, y_pred, zero_division=0)),
		"auc": float(roc_auc_score(y_true, y_prob)),
		"confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
	}


def print_vertical_metrics(title, metrics):
	print(f"\n===== {title} =====")
	label_width = max(len(str(key)) for key in metrics.keys())
	for key, value in metrics.items():
		print(f"{str(key):<{label_width}} : {value}")


def drop_unnamed_columns(df):
	unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
	if unnamed_cols:
		df = df.drop(columns=unnamed_cols)
	return df


def main():
	parser = argparse.ArgumentParser(description="Small DNN quick validation")
	parser.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Fold index for quick validation")
	args = parser.parse_args()

	project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
	data_dir = os.path.join(project_root, "data", "processed", "five_folds")
	target_col = "SeriousDlqin2yrs"

	train_path = os.path.join(data_dir, f"fold_{args.fold}_train.csv")
	val_path = os.path.join(data_dir, f"fold_{args.fold}_val.csv")

	if not os.path.exists(train_path):
		raise FileNotFoundError(f"找不到训练集文件: {train_path}")
	if not os.path.exists(val_path):
		raise FileNotFoundError(f"找不到验证集文件: {val_path}")

	train_df = drop_unnamed_columns(pd.read_csv(train_path))
	val_df = drop_unnamed_columns(pd.read_csv(val_path))

	if target_col not in train_df.columns or target_col not in val_df.columns:
		raise ValueError(f"目标列 {target_col} 不存在，请检查数据文件列名。")

	X_train = train_df.drop(columns=[target_col])
	y_train = train_df[target_col].astype(int)
	X_val = val_df.drop(columns=[target_col])
	y_val = val_df[target_col].astype(int)

	if list(X_train.columns) != list(X_val.columns):
		missing_cols = [c for c in X_train.columns if c not in X_val.columns]
		extra_cols = [c for c in X_val.columns if c not in X_train.columns]
		if missing_cols or extra_cols:
			raise ValueError(
				f"特征列不一致。缺失列: {missing_cols}, 多余列: {extra_cols}"
			)
		X_val = X_val[X_train.columns]

	# 单隐藏层网络：约 160 个神经元，快速验证
	model = Pipeline(
		steps=[
			("sc", StandardScaler()),
			(
				"mlp",
				MLPClassifier(
					hidden_layer_sizes=(160,),
					activation="relu",
					solver="adam",
					alpha=1e-4,
					learning_rate_init=1e-3,
					learning_rate="adaptive",
					max_iter=10,
					batch_size=32,
					random_state=42,
					early_stopping=True,
					n_iter_no_change=5,
					validation_fraction=0.1,
				),
			),
		]
	)

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		start_time = time.perf_counter()
		model.fit(X_train, y_train)
		train_time_seconds = time.perf_counter() - start_time

	y_prob = model.predict_proba(X_val)[:, 1]
	y_pred = (y_prob >= 0.5).astype(int)

	metrics = evaluate_binary(y_val, y_pred, y_prob)
	metrics["train_time_seconds"] = float(train_time_seconds)
	metrics["fold"] = args.fold

	print("\n===== DNN Small Quick Validation =====")
	print(f"Fold: {args.fold}")
	print_vertical_metrics("Validation Metrics", metrics)


if __name__ == "__main__":
	main()

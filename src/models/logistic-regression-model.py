import os
import time
import warnings
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	roc_auc_score,
	confusion_matrix,
)


def evaluate_binary(y_true, y_pred, y_prob):
	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"f1": float(f1_score(y_true, y_pred, zero_division=0)),
		"auc": float(roc_auc_score(y_true, y_prob)),
		"confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
	}


def build_prediction_frame(fold, val_df, y_true, y_pred, y_prob):
	return pd.DataFrame(
		{
			"fold": fold,
			"sample_index": val_df.index,
			"y_true": y_true.to_numpy(),
			"y_pred": y_pred,
			"y_prob": y_prob,
		}
	)


def print_vertical_metrics(title, metrics):
	print(f"\n===== {title} =====")
	label_width = max(len(str(key)) for key in metrics.keys())
	for key, value in metrics.items():
		print(f"{str(key):<{label_width}} : {value}")


def main():
	project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
	train_dir = os.path.join(project_root, "data", "processed", "five_folds_oversampled")
	val_dir = os.path.join(project_root, "data", "processed", "five_folds_standardized")

	target_col = "SeriousDlqin2yrs"

	# 逻辑回归参数：适合二分类+中等规模数据
	lr_params = {
		"penalty": "l2",
		"C": 1.0,
		"solver": "lbfgs",
		"max_iter": 2000,
		"random_state": 42,
	}

	fold_metrics = []
	oof_predictions = []

	for fold in range(1, 6):
		train_path = os.path.join(train_dir, f"fold_{fold}_train_oversampled.csv")
		val_path = os.path.join(val_dir, f"fold_{fold}_val_scaled.csv")

		if not os.path.exists(train_path):
			raise FileNotFoundError(f"找不到训练集文件: {train_path}")
		if not os.path.exists(val_path):
			raise FileNotFoundError(f"找不到验证集文件: {val_path}")

		# 训练集文件无索引列，验证集文件包含索引列
		train_df = pd.read_csv(train_path)
		val_df = pd.read_csv(val_path, index_col=0)

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
					f"第{fold}折特征列不一致。缺失列: {missing_cols}, 多余列: {extra_cols}"
				)
			X_val = X_val[X_train.columns]

		model = LogisticRegression(**lr_params)

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			start_time = time.perf_counter()
			model.fit(X_train, y_train)
			train_time_seconds = time.perf_counter() - start_time

		y_pred = model.predict(X_val)
		y_prob = model.predict_proba(X_val)[:, 1]

		metrics = evaluate_binary(y_val, y_pred, y_prob)
		metrics["fold"] = fold
		metrics["train_time_seconds"] = float(train_time_seconds)
		fold_metrics.append(metrics)

		fold_prediction_df = build_prediction_frame(fold, val_df, y_val, y_pred, y_prob)
		oof_predictions.append(fold_prediction_df)

	metrics_df = pd.DataFrame(fold_metrics)
	oof_predictions_df = pd.concat(oof_predictions, ignore_index=True)
	overall_metrics = evaluate_binary(
		oof_predictions_df["y_true"],
		oof_predictions_df["y_pred"],
		oof_predictions_df["y_prob"],
	)

	print("\n===== 5-Fold Metrics Table =====")
	print(metrics_df[["fold", "accuracy", "precision", "recall", "f1", "auc", "train_time_seconds"]].to_string(index=False))

	print_vertical_metrics("Overall OOF Metrics", overall_metrics)
	
	


if __name__ == "__main__":
	main()

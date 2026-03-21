import argparse
import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
	roc_curve,
)

try:
	import tensorflow as tf
	from tensorflow.keras import Sequential
	from tensorflow.keras.initializers import RandomUniform
	from tensorflow.keras.layers import Dense, Input
	from tensorflow.keras.optimizers import Adadelta
except ImportError as exc:
	raise ImportError(
		"未检测到 TensorFlow。请先安装 tensorflow 后再运行 DNN.py。"
	) from exc


def set_seed(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)


def drop_unnamed_columns(df):
	unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
	if unnamed_cols:
		df = df.drop(columns=unnamed_cols)
	return df


def evaluate_binary(y_true, y_pred, y_prob):
	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"f1": float(f1_score(y_true, y_pred, zero_division=0)),
		"auc": float(roc_auc_score(y_true, y_prob)),
		"confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
	}


def find_optimal_threshold_by_youden(y_true, y_prob):
	"""使用约登指数（Youden Index）找最优阈值。
	约登指数 J = 灵敏度 + 特异度 - 1 = recall + specificity - 1
	"""
	fpr, tpr, thresholds = roc_curve(y_true, y_prob)
	# 计算specificity（特异度）= 1 - FPR
	specificity = 1 - fpr
	# 计算约登指数
	youden_index = tpr + specificity - 1
	# 找到最大值对应的索引
	optimal_idx = np.argmax(youden_index)
	optimal_threshold = thresholds[optimal_idx]
	optimal_youden = youden_index[optimal_idx]
	
	return optimal_threshold, optimal_youden, float(tpr[optimal_idx]), float(specificity[optimal_idx])


def evaluate_with_custom_threshold(y_true, y_prob, threshold):
	"""使用自定义阈值评估模型。"""
	y_pred = (y_prob >= threshold).astype(int)
	return evaluate_binary(y_true, y_pred, y_prob)


def print_vertical_metrics(title, metrics):
	print(f"\n===== {title} =====")
	label_width = max(len(str(key)) for key in metrics.keys())
	for key, value in metrics.items():
		print(f"{str(key):<{label_width}} : {value}")


def build_dnn(input_dim):
	# 图片参数对应配置：
	# - 隐藏层激活: ReLU
	# - 输出层激活: Sigmoid（二分类）
	# - 权重初始化: 均匀分布[0, 0.05]
	# - 损失函数: Binary Cross-Entropy
	# - 优化器: 自适应学习率，rho=0.99, epsilon=1e-8
	kernel_init = RandomUniform(minval=0.0, maxval=0.05, seed=42)

	model = Sequential(
		[
			Input(shape=(input_dim,)),
			Dense(175, activation="relu", kernel_initializer=kernel_init),
			Dense(350, activation="relu", kernel_initializer=kernel_init),
			Dense(150, activation="relu", kernel_initializer=kernel_init),
			Dense(1, activation="sigmoid", kernel_initializer=kernel_init),
		]
	)

	optimizer = Adadelta(rho=0.99, epsilon=1e-8)
	model.compile(optimizer=optimizer, loss="binary_crossentropy")
	return model


def parse_folds(folds_text):
	folds = []
	for token in folds_text.split(","):
		token = token.strip()
		if not token:
			continue
		fold = int(token)
		if fold not in [1, 2, 3, 4, 5]:
			raise ValueError(f"非法折号: {fold}，仅支持 1-5")
		folds.append(fold)
	if not folds:
		raise ValueError("fold 列表为空，请通过 --folds 传入如 1,2,3")
	return folds


def main():
	parser = argparse.ArgumentParser(description="DNN model based on report settings")
	parser.add_argument("--folds", type=str, default="1,2,3,4,5", help="使用哪些折，格式如 1,2,3")
	parser.add_argument("--epochs", type=int, default=10, help="训练轮数，默认10")
	parser.add_argument("--batch_size", type=int, default=32, help="批大小，默认32")
	args = parser.parse_args()

	set_seed(42)
	folds = parse_folds(args.folds)

	project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
	data_dir = os.path.join(project_root, "data", "processed", "five_folds")
	target_col = "SeriousDlqin2yrs"

	fold_metrics = []
	oof_predictions = []

	for fold in folds:
		train_path = os.path.join(data_dir, f"fold_{fold}_train.csv")
		val_path = os.path.join(data_dir, f"fold_{fold}_val.csv")

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
					f"第{fold}折特征列不一致。缺失列: {missing_cols}, 多余列: {extra_cols}"
				)
			X_val = X_val[X_train.columns]

		input_dim = X_train.shape[1]
		if input_dim != 322:
			print(f"[提示] 当前输入维度为 {input_dim}，与报告中的 322 不一致。")

		model = build_dnn(input_dim=input_dim)

		start_time = time.perf_counter()
		model.fit(
			X_train.values,
			y_train.values,
			epochs=args.epochs,
			batch_size=args.batch_size,
			verbose=0,
		)
		train_time_seconds = time.perf_counter() - start_time

		y_prob = model.predict(X_val.values, verbose=0).reshape(-1)
		
		# 使用约登指数找最优阈值
		optimal_threshold, optimal_youden, optimal_tpr, optimal_specificity = find_optimal_threshold_by_youden(y_val.values, y_prob)
		y_pred = (y_prob >= optimal_threshold).astype(int)

		metrics = evaluate_binary(y_val, y_pred, y_prob)
		metrics["fold"] = fold
		metrics["train_time_seconds"] = float(train_time_seconds)
		metrics["optimal_threshold"] = float(optimal_threshold)
		metrics["youden_index"] = float(optimal_youden)
		metrics["optimal_tpr"] = float(optimal_tpr)
		metrics["optimal_specificity"] = float(optimal_specificity)
		fold_metrics.append(metrics)

		fold_prediction_df = pd.DataFrame(
			{
				"fold": fold,
				"sample_index": val_df.index,
				"y_true": y_val.to_numpy(),
				"y_pred": y_pred,
				"y_prob": y_prob,
				"optimal_threshold": optimal_threshold,
			}
		)
		oof_predictions.append(fold_prediction_df)

		print_vertical_metrics(f"Fold {fold} Metrics (Optimal Threshold={optimal_threshold:.4f})", metrics)

	metrics_df = pd.DataFrame(fold_metrics)
	oof_predictions_df = pd.concat(oof_predictions, ignore_index=True)
	
	# 对Overall预测使用所有fold的平均最优阈值
	average_threshold = metrics_df["optimal_threshold"].mean()
	overall_y_pred = (oof_predictions_df["y_prob"] >= average_threshold).astype(int)
	overall_metrics = evaluate_binary(
		oof_predictions_df["y_true"],
		overall_y_pred,
		oof_predictions_df["y_prob"],
	)
	overall_metrics["average_optimal_threshold"] = float(average_threshold)

	print("\n===== DNN Fold Metrics with Youden Index =====")
	print(
		metrics_df[
			["fold", "accuracy", "precision", "recall", "f1", "auc", "optimal_threshold", "youden_index", "train_time_seconds"]
		].to_string(index=False)
	)
	print_vertical_metrics("Overall OOF Metrics (with Average Optimal Threshold)", overall_metrics)


if __name__ == "__main__":
	main()

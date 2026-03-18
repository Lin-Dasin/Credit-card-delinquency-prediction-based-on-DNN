import os
import time
import pandas as pd

from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	roc_auc_score,
	confusion_matrix,
)

from xgboost import XGBClassifier


# ===================== 可配置参数区域 =====================
TARGET_COL = "SeriousDlqin2yrs"
RANDOM_STATE = 42
N_SPLITS = 5

# 目录配置：默认使用新增特征对应的 extra_five_folds 数据
TRAIN_DIR_NAME = "extra_five_folds_oversampled"
VAL_DIR_NAME = "extra_five_folds_standardized"
SOURCE_DATA_FILE_NAME = "train_set_processed_extra.csv"

# 列配置：
# 1) INCLUDE_FEATURE_COLUMNS 为空时，默认使用除目标列外的所有列
# 2) EXCLUDE_FEATURE_COLUMNS 中的列会被排除
# 3) INCLUDE_FEATURE_COLUMNS 非空时，只保留其中存在于数据内的列
INCLUDE_FEATURE_COLUMNS = []
EXCLUDE_FEATURE_COLUMNS = []
ENGINEERED_FEATURE_COLUMNS = [
	"IsUtilMaxed",
	"TotalLateTimes",
	"HighUtil_And_Late",
	"EstMonthlyDebt",
	"IncomePerDependent",
]

XGB_PARAMS = {
	"n_estimators": 300,
	"max_depth": 5,
	"learning_rate": 0.05,
	"subsample": 0.9,
	"colsample_bytree": 0.9,
	"objective": "binary:logistic",
	"eval_metric": "logloss",
	"random_state": RANDOM_STATE,
	"n_jobs": -1,
}
# =========================================================


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


def drop_unnamed_columns(df):
	"""移除历史 index 导出产生的 Unnamed 列，避免特征列错位。"""
	unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
	if unnamed_cols:
		df = df.drop(columns=unnamed_cols)
	return df


def print_feature_summary(selected_features):
	selected_set = set(selected_features)
	included_engineered = [
		column for column in ENGINEERED_FEATURE_COLUMNS if column in selected_set
	]
	missing_engineered = [
		column for column in ENGINEERED_FEATURE_COLUMNS if column not in selected_set
	]

	print(f"本次训练特征数: {len(selected_features)}")
	# print(f"本次训练特征: {selected_features}")
	# print(f"本次纳入的新增特征: {included_engineered}")
	if missing_engineered:
		print(f"[提示] 以下新增特征未参与本次训练: {missing_engineered}")


def explain_missing_engineered_features(project_root, selected_features):
	source_path = os.path.join(project_root, "data", "processed", SOURCE_DATA_FILE_NAME)
	if not os.path.exists(source_path):
		return

	source_columns = pd.read_csv(source_path, nrows=0).columns.tolist()
	source_feature_set = set(source_columns)
	selected_feature_set = set(selected_features)

	available_in_source = [
		column for column in ENGINEERED_FEATURE_COLUMNS if column in source_feature_set
	]
	missing_from_folds = [
		column
		for column in available_in_source
		if column not in selected_feature_set
	]

	if missing_from_folds:
		print(
			"[诊断] 这些新增特征已存在于 train_set_processed_extra.csv，但当前 fold 文件中没有它们: "
			f"{missing_from_folds}"
		)
		print("[诊断] 这通常说明你在新增特征后，还没有重新运行 scripts/test1.py 来生成最新的 extra_five_folds 数据。")


def resolve_feature_columns(train_df, val_df, target_col):
	if target_col not in train_df.columns or target_col not in val_df.columns:
		raise ValueError(f"目标列 {target_col} 不存在，请检查数据文件列名。")

	train_features = [c for c in train_df.columns if c != target_col]
	val_features = [c for c in val_df.columns if c != target_col]
	common_features = [c for c in train_features if c in val_features]

	if INCLUDE_FEATURE_COLUMNS:
		selected_features = [
			column
			for column in INCLUDE_FEATURE_COLUMNS
			if column in common_features and column not in EXCLUDE_FEATURE_COLUMNS
		]
		missing_included = [
			column
			for column in INCLUDE_FEATURE_COLUMNS
			if column not in common_features
		]
		if missing_included:
			print(f"[警告] 这些指定列不在本次折数据中，已跳过: {missing_included}")
	else:
		selected_features = [
			column for column in common_features if column not in EXCLUDE_FEATURE_COLUMNS
		]

	if not selected_features:
		raise ValueError("没有可用于训练的特征列，请检查 INCLUDE_FEATURE_COLUMNS / EXCLUDE_FEATURE_COLUMNS 配置。")

	missing_from_val = [column for column in selected_features if column not in val_features]
	if missing_from_val:
		raise ValueError(f"验证集中缺少必要特征列: {missing_from_val}")

	return selected_features


def load_fold_data(train_path, val_path, target_col):
	train_df = drop_unnamed_columns(pd.read_csv(train_path))
	val_df = drop_unnamed_columns(pd.read_csv(val_path))

	selected_features = resolve_feature_columns(train_df, val_df, target_col)

	X_train = train_df[selected_features]
	y_train = train_df[target_col].astype(int)
	X_val = val_df[selected_features]
	y_val = val_df[target_col].astype(int)

	return X_train, y_train, X_val, y_val, val_df


def main():
	project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
	train_dir = os.path.join(project_root, "data", "processed", TRAIN_DIR_NAME)
	val_dir = os.path.join(project_root, "data", "processed", VAL_DIR_NAME)

	fold_metrics = []
	oof_predictions = []
	fold_gain_scores = []
	feature_columns = None

	print(f"训练目录: {train_dir}")
	print(f"验证目录: {val_dir}")

	for fold in range(1, N_SPLITS + 1):
		train_path = os.path.join(train_dir, f"fold_{fold}_train_oversampled.csv")
		val_path = os.path.join(val_dir, f"fold_{fold}_val_scaled.csv")

		if not os.path.exists(train_path):
			raise FileNotFoundError(f"找不到训练集文件: {train_path}")
		if not os.path.exists(val_path):
			raise FileNotFoundError(f"找不到验证集文件: {val_path}")

		X_train, y_train, X_val, y_val, val_df = load_fold_data(
			train_path=train_path,
			val_path=val_path,
			target_col=TARGET_COL,
		)

		if feature_columns is None:
			feature_columns = X_train.columns.tolist()
			print_feature_summary(feature_columns)
			explain_missing_engineered_features(project_root, feature_columns)

		model = XGBClassifier(**XGB_PARAMS)
		start_time = time.perf_counter()
		model.fit(X_train, y_train)
		train_time_seconds = time.perf_counter() - start_time
		fold_gain_scores.append(model.get_booster().get_score(importance_type="gain"))

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

	print("\n===== 5-Fold Metrics Table  =====")
	print(
		metrics_df[
			["fold", "accuracy", "precision", "recall", "f1", "auc", "train_time_seconds"]
		].to_string(index=False)
	)

	print_vertical_metrics("Overall OOF Metrics IN XGBoost", overall_metrics)

	if fold_gain_scores:
		gain_df = pd.DataFrame(fold_gain_scores).fillna(0.0)
		avg_gain_series = gain_df.mean(axis=0).sort_values(ascending=False)
		avg_gain_df = avg_gain_series.reset_index()
		avg_gain_df.columns = ["feature", "avg_gain"]

		print("\n===== Average Gain Importance (Across Folds) =====")
		print(avg_gain_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
		print(f"\n整体平均信息增益: {avg_gain_series.mean():.6f}")


if __name__ == "__main__":
	main()

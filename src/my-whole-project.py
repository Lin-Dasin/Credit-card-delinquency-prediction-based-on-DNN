import os
import random
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


SEED = 42
TARGET_COL = "SeriousDlqin2yrs"
TEST_SIZE = 0.2
N_SPLITS = 5
TARGET_MAJORITY_MINORITY_RATIO = (75, 25)
USE_SMOTE = True


def set_global_seed(seed=SEED):
	np.random.seed(seed)
	random.seed(seed)


def drop_unnamed_columns(df):
	unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
	if unnamed_cols:
		df = df.drop(columns=unnamed_cols)
	return df


def load_raw_data(project_root):
	raw_path = os.path.join(project_root, "data", "raw", "cs-training.csv")
	if not os.path.exists(raw_path):
		raise FileNotFoundError(f"Raw data file not found: {raw_path}")

	df_raw = pd.read_csv(raw_path)

	# Align with notebook behavior: detect and remove ID-like first column.
	first_col = df_raw.columns[0]
	first_col_series = df_raw[first_col]
	is_integer_like = pd.api.types.is_integer_dtype(first_col_series)
	is_unique = first_col_series.nunique(dropna=False) == len(df_raw)
	name_like_id = str(first_col).strip().lower() in {"id", "unnamed: 0"}
	if (name_like_id and is_unique) or (is_integer_like and is_unique):
		df_raw = df_raw.iloc[:, 1:]

	df_raw = drop_unnamed_columns(df_raw)
	return df_raw


def hard_clean(df):
	df_cleaned = df.copy()
	original_count = len(df_cleaned)

	# Remove invalid due-day encoded values.
	due_cols = [
		"NumberOfTime30-59DaysPastDueNotWorse",
		"NumberOfTime60-89DaysPastDueNotWorse",
		"NumberOfTimes90DaysLate",
	]
	mask_due = False
	for col in due_cols:
		mask_due |= (df_cleaned[col] == 96) | (df_cleaned[col] == 98)
	due_delete_count = int(mask_due.sum())
	df_cleaned = df_cleaned[~mask_due]

	mask_age = df_cleaned["age"] < 18
	age_delete_count = int(mask_age.sum())
	df_cleaned = df_cleaned[~mask_age]

	mask_income = df_cleaned["MonthlyIncome"] > 1_000_000
	income_delete_count = int(mask_income.sum())
	df_cleaned = df_cleaned[~mask_income]

	total_deleted = due_delete_count + age_delete_count + income_delete_count
	summary = {
		"original_rows": original_count,
		"cleaned_rows": len(df_cleaned),
		"deleted_total": total_deleted,
		"deleted_due_96_98": due_delete_count,
		"deleted_age_lt_18": age_delete_count,
		"deleted_income_gt_1m": income_delete_count,
	}
	return df_cleaned.reset_index(drop=True), summary


def fit_train_processing_params(train_df):
	monthly_income_nonzero = train_df.loc[train_df["MonthlyIncome"] > 0, "MonthlyIncome"]
	income_median = float(monthly_income_nonzero.median())

	params = {
		"monthly_income_fill": income_median,
		"dependents_fill": 0.0,
		"debt_ratio_clip": (0.0, 10.0),
		"util_clip": (0.0, 1.0),
	}
	return params


def apply_processing(df, params):
	out = df.copy()

	out["MonthlyIncome"] = out["MonthlyIncome"].fillna(params["monthly_income_fill"]).replace(
		0, params["monthly_income_fill"]
	)
	out["NumberOfDependents"] = out["NumberOfDependents"].fillna(params["dependents_fill"])

	debt_low, debt_high = params["debt_ratio_clip"]
	util_low, util_high = params["util_clip"]
	out["DebtRatio"] = out["DebtRatio"].clip(lower=debt_low, upper=debt_high)
	out["RevolvingUtilizationOfUnsecuredLines"] = out[
		"RevolvingUtilizationOfUnsecuredLines"
	].clip(lower=util_low, upper=util_high)

	return out


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


def oversample_train(X_train, y_train, target_ratio, random_state=SEED, use_smote=USE_SMOTE):
	try:
		from imblearn.over_sampling import SMOTE

		smote_available = True
	except ImportError:
		smote_available = False

	if use_smote and smote_available:
		sampler = SMOTE(sampling_strategy=target_ratio, random_state=random_state)
		X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
		method = "SMOTE"
	else:
		train_df = pd.DataFrame(X_train).copy()
		train_df[TARGET_COL] = y_train

		class_counts = train_df[TARGET_COL].value_counts()
		majority_label = class_counts.idxmax()
		minority_label = class_counts.idxmin()

		majority_df = train_df[train_df[TARGET_COL] == majority_label]
		minority_df = train_df[train_df[TARGET_COL] == minority_label]

		target_minority_count = int(len(majority_df) * target_ratio)
		target_minority_count = max(target_minority_count, len(minority_df))

		minority_upsampled = resample(
			minority_df,
			replace=True,
			n_samples=target_minority_count,
			random_state=random_state,
		)
		out_df = pd.concat([majority_df, minority_upsampled], axis=0).sample(
			frac=1.0, random_state=random_state
		)

		X_resampled = out_df.drop(columns=[TARGET_COL]).to_numpy()
		y_resampled = out_df[TARGET_COL].to_numpy()
		method = "RandomOverSampling"

	return X_resampled, y_resampled, method


def run_5fold_logistic(train_df):
	X = train_df.drop(columns=[TARGET_COL])
	y = train_df[TARGET_COL].astype(int)

	skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
	target_ratio = TARGET_MAJORITY_MINORITY_RATIO[1] / TARGET_MAJORITY_MINORITY_RATIO[0]

	lr_params = {
		"penalty": "l2",
		"C": 1.0,
		"solver": "lbfgs",
		"max_iter": 2000,
		"random_state": SEED,
	}

	fold_metrics = []
	oof_predictions = []
	method_counter = Counter()

	for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
		X_train_fold = X.iloc[train_idx]
		y_train_fold = y.iloc[train_idx]
		X_val_fold = X.iloc[val_idx]
		y_val_fold = y.iloc[val_idx]

		scaler = StandardScaler()
		X_train_scaled = scaler.fit_transform(X_train_fold)
		X_val_scaled = scaler.transform(X_val_fold)

		X_resampled, y_resampled, method = oversample_train(
			X_train_scaled,
			y_train_fold.to_numpy(),
			target_ratio=target_ratio,
			random_state=SEED,
			use_smote=USE_SMOTE,
		)
		method_counter[method] += 1

		model = LogisticRegression(**lr_params)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			start_time = time.perf_counter()
			model.fit(X_resampled, y_resampled)
			train_time_seconds = time.perf_counter() - start_time

		y_pred = model.predict(X_val_scaled)
		y_prob = model.predict_proba(X_val_scaled)[:, 1]

		metrics = evaluate_binary(y_val_fold, y_pred, y_prob)
		metrics["fold"] = fold
		metrics["train_time_seconds"] = float(train_time_seconds)
		metrics["oversample_method"] = method
		fold_metrics.append(metrics)

		pred_df = pd.DataFrame(
			{
				"fold": fold,
				"sample_index": y_val_fold.index,
				"y_true": y_val_fold.to_numpy(),
				"y_pred": y_pred,
				"y_prob": y_prob,
			}
		)
		oof_predictions.append(pred_df)

	metrics_df = pd.DataFrame(fold_metrics)
	oof_predictions_df = pd.concat(oof_predictions, ignore_index=True)
	overall_metrics = evaluate_binary(
		oof_predictions_df["y_true"],
		oof_predictions_df["y_pred"],
		oof_predictions_df["y_prob"],
	)

	print("\n===== 5-Fold Metrics Table =====")
	print(
		metrics_df[
			[
				"fold",
				"accuracy",
				"precision",
				"recall",
				"f1",
				"auc",
				"train_time_seconds",
				"oversample_method",
			]
		].to_string(index=False)
	)
	print(f"\nOversampling method usage by fold: {dict(method_counter)}")
	print_vertical_metrics("Overall OOF Metrics", overall_metrics)

	return metrics_df, overall_metrics


def run_holdout_logistic(train_df, test_df):
	X_train = train_df.drop(columns=[TARGET_COL])
	y_train = train_df[TARGET_COL].astype(int)
	X_test = test_df.drop(columns=[TARGET_COL])
	y_test = test_df[TARGET_COL].astype(int)

	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	target_ratio = TARGET_MAJORITY_MINORITY_RATIO[1] / TARGET_MAJORITY_MINORITY_RATIO[0]
	X_resampled, y_resampled, method = oversample_train(
		X_train_scaled,
		y_train.to_numpy(),
		target_ratio=target_ratio,
		random_state=SEED,
		use_smote=USE_SMOTE,
	)

	model = LogisticRegression(
		penalty="l2",
		C=1.0,
		solver="lbfgs",
		max_iter=2000,
		random_state=SEED,
	)

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		model.fit(X_resampled, y_resampled)

	y_pred = model.predict(X_test_scaled)
	y_prob = model.predict_proba(X_test_scaled)[:, 1]
	holdout_metrics = evaluate_binary(y_test, y_pred, y_prob)

	print(f"\nHoldout oversampling method: {method}")
	print_vertical_metrics("Holdout Test Metrics", holdout_metrics)

	return holdout_metrics


def main():
	warnings.filterwarnings("ignore", category=FutureWarning)
	warnings.filterwarnings("ignore", category=UserWarning)
	set_global_seed(SEED)

	project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	print(f"Project root: {project_root}")

	df_raw = load_raw_data(project_root)
	print(f"Raw data shape: {df_raw.shape}")

	df_cleaned, clean_summary = hard_clean(df_raw)
	print("\n===== Hard Cleaning Summary =====")
	for k, v in clean_summary.items():
		print(f"{k}: {v}")

	X = df_cleaned.drop(columns=[TARGET_COL])
	y = df_cleaned[TARGET_COL].astype(int)

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=TEST_SIZE,
		random_state=SEED,
		stratify=y,
	)

	train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
	test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

	print("\n===== Stratified Split Summary =====")
	print(f"Train rows: {len(train_df)}, positive rate: {train_df[TARGET_COL].mean():.4f}")
	print(f"Test  rows: {len(test_df)}, positive rate: {test_df[TARGET_COL].mean():.4f}")

	process_params = fit_train_processing_params(train_df)
	train_processed = apply_processing(train_df, process_params)
	test_processed = apply_processing(test_df, process_params)

	print("\n===== Processing Params (fit on train only) =====")
	for k, v in process_params.items():
		print(f"{k}: {v}")

	run_5fold_logistic(train_processed)
	run_holdout_logistic(train_processed, test_processed)


if __name__ == "__main__":
	main()

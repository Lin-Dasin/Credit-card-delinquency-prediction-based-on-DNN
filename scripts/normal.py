import argparse
import ast
import os
import re

import pandas as pd
from sklearn.preprocessing import StandardScaler


def drop_unnamed_columns(df):
	unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
	if unnamed_cols:
		df = df.drop(columns=unnamed_cols)
	return df


def parse_param_dict(raw_text):
	"""解析参数文件中的字典字符串，兼容 np.float64(x) 形式。"""
	if pd.isna(raw_text):
		return {}
	text = str(raw_text).strip()
	if not text:
		return {}
	# train_process_params.csv 中可能出现 np.float64(5447.5)
	text = re.sub(r"np\.float64\(([^\)]+)\)", r"\1", text)
	return ast.literal_eval(text)


def process_test_set_with_train_params(test_df, params_path):
	if not os.path.exists(params_path):
		raise FileNotFoundError(f"找不到训练处理参数文件: {params_path}")

	params_df = pd.read_csv(params_path)
	if params_df.empty:
		raise ValueError("训练处理参数文件为空，无法处理测试集。")

	row = params_df.iloc[0]
	missing_fill = parse_param_dict(row.get("missing_fill", "{}"))
	clip_rules = parse_param_dict(row.get("clip_rules", "{}"))

	processed = test_df.copy()

	for key, value in missing_fill.items():
		if key.endswith("_median"):
			col = key[: -len("_median")]
		elif key.endswith("_fill"):
			col = key[: -len("_fill")]
		else:
			col = key

		if col in processed.columns:
			processed[col] = processed[col].fillna(value)

	for key, value in clip_rules.items():
		if not key.endswith("_clip_range"):
			continue
		col = key[: -len("_clip_range")]
		if col in processed.columns and isinstance(value, (list, tuple)) and len(value) == 2:
			processed[col] = processed[col].clip(lower=value[0], upper=value[1])

	return processed


def main():
	parser = argparse.ArgumentParser(description="标准化总训练集和测试集")
	parser.add_argument(
		"--train_path",
		type=str,
		default=os.path.join("data", "processed", "train_set_processed.csv"),
		help="训练集路径（建议为已处理后的 train_set_processed.csv）",
	)
	parser.add_argument(
		"--test_path",
		type=str,
		default=os.path.join("data", "processed", "test_set.csv"),
		help="测试集原始路径（将依据训练参数做填充+截尾）",
	)
	parser.add_argument(
		"--train_params_path",
		type=str,
		default=os.path.join("data", "processed", "train_process_params.csv"),
		help="训练集处理参数文件路径",
	)
	parser.add_argument(
		"--target_col",
		type=str,
		default="SeriousDlqin2yrs",
		help="目标列列名",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default=os.path.join("data", "processed"),
		help="输出目录",
	)
	args = parser.parse_args()

	if not os.path.exists(args.train_path):
		raise FileNotFoundError(f"找不到训练集文件: {args.train_path}")
	if not os.path.exists(args.test_path):
		raise FileNotFoundError(f"找不到测试集文件: {args.test_path}")

	os.makedirs(args.output_dir, exist_ok=True)

	train_df = drop_unnamed_columns(pd.read_csv(args.train_path))
	raw_test_df = drop_unnamed_columns(pd.read_csv(args.test_path))

	# 1) 先按训练参数处理测试集（填充 + 截尾）
	test_df = process_test_set_with_train_params(raw_test_df, args.train_params_path)
	test_processed_output_path = os.path.join(args.output_dir, "test_set_processed.csv")
	test_df.to_csv(test_processed_output_path, index=False)

	train_has_target = args.target_col in train_df.columns
	test_has_target = args.target_col in test_df.columns

	if train_has_target:
		x_train = train_df.drop(columns=[args.target_col])
		y_train = train_df[args.target_col]
	else:
		x_train = train_df.copy()
		y_train = None

	if test_has_target:
		x_test = test_df.drop(columns=[args.target_col])
		y_test = test_df[args.target_col]
	else:
		x_test = test_df.copy()
		y_test = None

	if list(x_train.columns) != list(x_test.columns):
		raise ValueError("训练集与测试集特征列不一致，请先对齐列名和顺序。")

	scaler = StandardScaler()
	x_train_scaled = scaler.fit_transform(x_train)
	x_test_scaled = scaler.transform(x_test)

	train_scaled_df = pd.DataFrame(x_train_scaled, columns=x_train.columns)
	test_scaled_df = pd.DataFrame(x_test_scaled, columns=x_test.columns)

	if y_train is not None:
		train_scaled_df.insert(0, args.target_col, y_train.to_numpy())
	if y_test is not None:
		test_scaled_df.insert(0, args.target_col, y_test.to_numpy())

	scaler_params_df = pd.DataFrame(
		{
			"feature": x_train.columns,
			"mean": scaler.mean_,
			"scale": scaler.scale_,
			"var": scaler.var_,
		}
	)

	train_output_path = os.path.join(args.output_dir, "train_set_standardized.csv")
	test_output_path = os.path.join(args.output_dir, "test_set_standardized.csv")
	params_output_path = os.path.join(args.output_dir, "train_set_scaler_params.csv")

	train_scaled_df.to_csv(train_output_path, index=False)
	test_scaled_df.to_csv(test_output_path, index=False)
	scaler_params_df.to_csv(params_output_path, index=False)

	print("标准化完成。")
	print(f"测试集处理后文件: {test_processed_output_path}")
	print(f"训练集标准化文件: {train_output_path}")
	print(f"测试集标准化文件: {test_output_path}")
	print(f"训练集标准化参数: {params_output_path}")


if __name__ == "__main__":
	main()

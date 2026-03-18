import argparse
import os
import pandas as pd


LATE_30_59_COL = "NumberOfTime30-59DaysPastDueNotWorse"
LATE_60_89_COL = "NumberOfTime60-89DaysPastDueNotWorse"
LATE_90_COL = "NumberOfTimes90DaysLate"

def _resolve_input_path(explicit_input: str | None) -> str:
	if explicit_input:
		if not os.path.exists(explicit_input):
			raise FileNotFoundError(f"指定输入文件不存在: {explicit_input}")
		return os.path.abspath(explicit_input)

	candidate_paths = [
		"../data/processed/train_set_processed.csv",
		"data/processed/train_set_processed.csv",
	]

	for path in candidate_paths:
		if os.path.exists(path):
			return os.path.abspath(path)

	raise FileNotFoundError("未找到 train_set_processed.csv，请检查数据路径。")


def _resolve_output_path(input_path: str, explicit_output: str | None) -> str:
	if explicit_output:
		return os.path.abspath(explicit_output)

	input_dir = os.path.dirname(input_path)
	return os.path.join(input_dir, "train_set_processed_extra.csv")


def _resolve_util_column(df: pd.DataFrame) -> str:
	candidates = [
		"RevolvingUtilization",
		"RevolvingUtilizationOfUnsecuredLines",
	]

	for column in candidates:
		if column in df.columns:
			return column

	raise ValueError(
		"找不到额度使用率列，请确认输入文件中包含 RevolvingUtilization 或 RevolvingUtilizationOfUnsecuredLines。"
	)


def _validate_late_columns(df: pd.DataFrame) -> list[str]:
	late_columns = [LATE_30_59_COL, LATE_60_89_COL, LATE_90_COL]
	missing_columns = [column for column in late_columns if column not in df.columns]
	if missing_columns:
		raise ValueError(f"缺少逾期字段: {missing_columns}")
	return late_columns


def add_is_util_maxed(df: pd.DataFrame) -> pd.DataFrame:
	util_column = _resolve_util_column(df)
	util_series = pd.to_numeric(df[util_column], errors="coerce")
	df["IsUtilMaxed"] = util_series.isin([1.0, 0.9999]).astype("int8")
	return df


def add_total_late_times(df: pd.DataFrame) -> pd.DataFrame:
	late_columns = _validate_late_columns(df)
	df["TotalLateTimes"] = df[late_columns].sum(axis=1).astype("int64")
	return df


def add_late_severity_score(df: pd.DataFrame) -> pd.DataFrame:
	_validate_late_columns(df)
	df["LateSeverityScore"] = (
		df[LATE_30_59_COL] * 1
		+ df[LATE_60_89_COL] * 2
		+ df[LATE_90_COL] * 3
	).astype("int64")
	return df


def add_high_util_and_late(df: pd.DataFrame) -> pd.DataFrame:
	util_column = _resolve_util_column(df)
	if "TotalLateTimes" not in df.columns:
		raise ValueError("TotalLateTimes 列不存在，请先调用 add_total_late_times()")

	util_series = pd.to_numeric(df[util_column], errors="coerce")
	df["HighUtil_And_Late"] = (
		(util_series > 0.8) & (df["TotalLateTimes"] > 0)
	).astype("int8")
	return df


def add_utilization_x_total_late(df: pd.DataFrame) -> pd.DataFrame:
	util_column = _resolve_util_column(df)
	if "TotalLateTimes" not in df.columns:
		raise ValueError("TotalLateTimes 列不存在，请先调用 add_total_late_times()")

	util_series = pd.to_numeric(df[util_column], errors="coerce")
	df["Utilization_x_TotalLate"] = (util_series * df["TotalLateTimes"]).astype("float64")
	return df


def add_est_monthly_debt(df: pd.DataFrame) -> pd.DataFrame:
	if "MonthlyIncome" not in df.columns or "DebtRatio" not in df.columns:
		raise ValueError("缺少必要列：MonthlyIncome 或 DebtRatio")

	income_series = pd.to_numeric(df["MonthlyIncome"], errors="coerce")
	debt_ratio_series = pd.to_numeric(df["DebtRatio"], errors="coerce")
	
	# 处理缺失值：使用0代替NaN（表示无收入或无债务）
	income_series = income_series.fillna(0)
	debt_ratio_series = debt_ratio_series.fillna(0)
	
	df["EstMonthlyDebt"] = (income_series * debt_ratio_series).astype("float64")
	return df


def add_income_per_dependent(df: pd.DataFrame) -> pd.DataFrame:
	if "MonthlyIncome" not in df.columns or "NumberOfDependents" not in df.columns:
		raise ValueError("缺少必要列：MonthlyIncome 或 NumberOfDependents")

	income_series = pd.to_numeric(df["MonthlyIncome"], errors="coerce")
	dependents_series = pd.to_numeric(df["NumberOfDependents"], errors="coerce")
	
	# 处理缺失值
	income_series = income_series.fillna(0)
	dependents_series = dependents_series.fillna(0)
	
	# IncomePerDependent = MonthlyIncome / (NumberOfDependents + 1)
	df["IncomePerDependent"] = (income_series / (dependents_series + 1)).astype("float64")
	return df


def _save_csv(df: pd.DataFrame, output_path: str) -> None:
	output_dir = os.path.dirname(output_path)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)

	temp_path = f"{output_path}.tmp"
	df.to_csv(temp_path, index=True)
	try:
		os.replace(temp_path, output_path)
	except PermissionError as exc:
		if os.path.exists(temp_path):
			os.remove(temp_path)
		raise PermissionError(
			f"无法覆盖文件: {output_path}。请先关闭占用该文件的程序（如 Excel、Notebook 预览或其他编辑器）。"
		) from exc


def main() -> None:
	parser = argparse.ArgumentParser(
		description="读取 train_set_processed.csv，生成特征并保存到 train_set_processed_extra.csv"
	)
	parser.add_argument(
		"--input",
		type=str,
		default=None,
		help="输入 CSV 路径（可选，默认使用 train_set_processed.csv）",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=None,
		help="输出 CSV 路径（可选，默认使用 train_set_processed_extra.csv）",
	)
	args = parser.parse_args()

	try:
		input_path = _resolve_input_path(args.input)
		output_path = _resolve_output_path(input_path, args.output)
		print(f"输入文件: {input_path}")
		print(f"输出文件: {output_path}")

		df = pd.read_csv(input_path, index_col=0)

		df = add_is_util_maxed(df)
		df = add_total_late_times(df)
		# df = add_late_severity_score(df)
		df = add_high_util_and_late(df)
		# df = add_utilization_x_total_late(df)
		df = add_est_monthly_debt(df)
		df = add_income_per_dependent(df)

		_save_csv(df, output_path)

		print("已生成列: IsUtilMaxed, TotalLateTimes, HighUtil_And_Late, EstMonthlyDebt, IncomePerDependent")
		print("处理完成")

	except Exception as exc:
		print(f"处理失败: {type(exc).__name__}: {exc}")
		raise


if __name__ == "__main__":
	main()

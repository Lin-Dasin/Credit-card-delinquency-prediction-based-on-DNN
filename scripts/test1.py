import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


# ===================== 可配置区域 =====================
TARGET_COL = "SeriousDlqin2yrs"
RANDOM_STATE = 42
N_SPLITS = 5
INPUT_FILE_NAME = "train_set_processed_extra.csv"
RAW_OUTPUT_DIR_NAME = "extra_five_folds"
STANDARDIZED_OUTPUT_DIR_NAME = "extra_five_folds_standardized"
OVERSAMPLED_OUTPUT_DIR_NAME = "extra_five_folds_oversampled"

# 后续删列或不想参与导出的列，直接改这里。
DROP_COLUMNS = []

# 过采样目标比例：多数类 : 少数类 = 75 : 25
TARGET_MAJORITY_MINORITY_RATIO = (75, 25)
USE_SMOTE = True
# ====================================================


def resolve_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_input_path(project_root: str) -> str:
    candidate_paths = [
        os.path.join(project_root, "data", "processed", INPUT_FILE_NAME),
        os.path.join(project_root, "data", "processed", "train_set_processed.csv"),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("未找到 train_set_processed_extra.csv 或 train_set_processed.csv。")


def build_output_dirs(project_root: str) -> dict:
    processed_dir = os.path.join(project_root, "data", "processed")
    raw_dir = os.path.join(processed_dir, RAW_OUTPUT_DIR_NAME)
    standardized_dir = os.path.join(processed_dir, STANDARDIZED_OUTPUT_DIR_NAME)
    oversampled_dir = os.path.join(processed_dir, OVERSAMPLED_OUTPUT_DIR_NAME)

    for path in [raw_dir, standardized_dir, oversampled_dir]:
        os.makedirs(path, exist_ok=True)

    return {
        "processed": processed_dir,
        "raw": raw_dir,
        "standardized": standardized_dir,
        "oversampled": oversampled_dir,
    }


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COL not in df.columns:
        raise ValueError(f"未找到目标列: {TARGET_COL}")

    missing_drop_columns = [column for column in DROP_COLUMNS if column not in df.columns]
    if missing_drop_columns:
        raise ValueError(f"这些待删除列不存在: {missing_drop_columns}")

    selected_columns = [
        column for column in df.columns if column == TARGET_COL or column not in DROP_COLUMNS
    ]
    return df[selected_columns].copy()


def save_raw_folds(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    summary = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        fold_train = df.iloc[train_idx].copy()
        fold_val = df.iloc[val_idx].copy()

        train_path = os.path.join(output_dir, f"fold_{fold}_train.csv")
        val_path = os.path.join(output_dir, f"fold_{fold}_val.csv")
        fold_train.to_csv(train_path, index=True)
        fold_val.to_csv(val_path, index=True)

        summary.append(
            {
                "fold": fold,
                "train_size": len(fold_train),
                "val_size": len(fold_val),
                "train_pos_rate": fold_train[TARGET_COL].mean(),
                "val_pos_rate": fold_val[TARGET_COL].mean(),
            }
        )

    return pd.DataFrame(summary)


def save_standardized_folds(raw_dir: str, output_dir: str) -> pd.DataFrame:
    summary = []

    for fold in range(1, N_SPLITS + 1):
        train_path = os.path.join(raw_dir, f"fold_{fold}_train.csv")
        val_path = os.path.join(raw_dir, f"fold_{fold}_val.csv")

        fold_train = pd.read_csv(train_path, index_col=0)
        fold_val = pd.read_csv(val_path, index_col=0)
        feature_columns = [column for column in fold_train.columns if column != TARGET_COL]

        scaler = StandardScaler()
        scaler.fit(fold_train[feature_columns])

        scaler_params = pd.DataFrame(
            {
                "feature": feature_columns,
                "mean": scaler.mean_,
                "std": scaler.scale_,
            }
        )
        scaler_params.to_csv(
            os.path.join(output_dir, f"fold_{fold}_scaler_params.csv"),
            index=False,
        )

        train_scaled = pd.DataFrame(
            scaler.transform(fold_train[feature_columns]),
            columns=feature_columns,
            index=fold_train.index,
        )
        val_scaled = pd.DataFrame(
            scaler.transform(fold_val[feature_columns]),
            columns=feature_columns,
            index=fold_val.index,
        )
        train_scaled[TARGET_COL] = fold_train[TARGET_COL].values
        val_scaled[TARGET_COL] = fold_val[TARGET_COL].values

        train_scaled.to_csv(os.path.join(output_dir, f"fold_{fold}_train_scaled.csv"), index=True)
        val_scaled.to_csv(os.path.join(output_dir, f"fold_{fold}_val_scaled.csv"), index=True)

        summary.append(
            {
                "fold": fold,
                "train_size": len(train_scaled),
                "val_size": len(val_scaled),
                "train_pos_rate": train_scaled[TARGET_COL].mean(),
                "val_pos_rate": val_scaled[TARGET_COL].mean(),
            }
        )

    return pd.DataFrame(summary)


def oversample_single_fold(
    fold_number: int,
    standardized_dir: str,
    output_dir: str,
    target_ratio: float,
) -> dict:
    train_path = os.path.join(standardized_dir, f"fold_{fold_number}_train_scaled.csv")
    train_df = pd.read_csv(train_path, index_col=0)

    feature_columns = [column for column in train_df.columns if column != TARGET_COL]
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COL]
    before_counts = y_train.value_counts().to_dict()

    method_used = "RandomOverSampling"
    if USE_SMOTE:
        try:
            from imblearn.over_sampling import SMOTE

            sampler = SMOTE(sampling_strategy=target_ratio, random_state=RANDOM_STATE)
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            method_used = "SMOTE"
        except ImportError:
            X_resampled, y_resampled = random_oversample(X_train, y_train, target_ratio)
    else:
        X_resampled, y_resampled = random_oversample(X_train, y_train, target_ratio)

    oversampled_df = pd.DataFrame(X_resampled, columns=feature_columns)
    oversampled_df[TARGET_COL] = y_resampled
    oversampled_df.to_csv(
        os.path.join(output_dir, f"fold_{fold_number}_train_oversampled.csv"),
        index=False,
    )

    after_counts = oversampled_df[TARGET_COL].value_counts().to_dict()
    return {
        "fold": fold_number,
        "method": method_used,
        "train_before_0": before_counts.get(0, 0),
        "train_before_1": before_counts.get(1, 0),
        "train_after_0": after_counts.get(0, 0),
        "train_after_1": after_counts.get(1, 0),
        "after_ratio_1_to_0": round(after_counts.get(1, 0) / max(after_counts.get(0, 1), 1), 4),
    }


def random_oversample(X_train: pd.DataFrame, y_train: pd.Series, target_ratio: float):
    train_joined = X_train.copy()
    train_joined[TARGET_COL] = y_train.values

    class_counts = train_joined[TARGET_COL].value_counts()
    majority_label = class_counts.idxmax()
    minority_label = class_counts.idxmin()

    majority_df = train_joined[train_joined[TARGET_COL] == majority_label]
    minority_df = train_joined[train_joined[TARGET_COL] == minority_label]

    target_minority_count = int(len(majority_df) * target_ratio)
    target_minority_count = max(target_minority_count, len(minority_df))

    minority_upsampled = resample(
        minority_df,
        replace=True,
        n_samples=target_minority_count,
        random_state=RANDOM_STATE,
    )

    resampled_df = pd.concat([majority_df, minority_upsampled], axis=0)
    resampled_df = resampled_df.sample(frac=1, random_state=RANDOM_STATE)

    X_resampled = resampled_df.drop(columns=[TARGET_COL])
    y_resampled = resampled_df[TARGET_COL]
    return X_resampled, y_resampled


def save_oversampled_folds(standardized_dir: str, output_dir: str) -> pd.DataFrame:
    target_ratio = TARGET_MAJORITY_MINORITY_RATIO[1] / TARGET_MAJORITY_MINORITY_RATIO[0]
    summary = []

    for fold in range(1, N_SPLITS + 1):
        summary.append(
            oversample_single_fold(
                fold_number=fold,
                standardized_dir=standardized_dir,
                output_dir=output_dir,
                target_ratio=target_ratio,
            )
        )

    return pd.DataFrame(summary)


def main() -> None:
    project_root = resolve_project_root()
    input_path = resolve_input_path(project_root)
    output_dirs = build_output_dirs(project_root)

    df = pd.read_csv(input_path, index_col=0)
    df = prepare_dataset(df)

    raw_summary = save_raw_folds(df, output_dirs["raw"])
    standardized_summary = save_standardized_folds(output_dirs["raw"], output_dirs["standardized"])
    oversampled_summary = save_oversampled_folds(output_dirs["standardized"], output_dirs["oversampled"])

    raw_summary.to_csv(os.path.join(output_dirs["raw"], "fold_summary.csv"), index=False)
    standardized_summary.to_csv(os.path.join(output_dirs["standardized"], "fold_summary.csv"), index=False)
    oversampled_summary.to_csv(os.path.join(output_dirs["oversampled"], "fold_summary.csv"), index=False)

    print(f"输入文件: {input_path}")
    print(f"原始折数据目录: {output_dirs['raw']}")
    print(f"标准化数据目录: {output_dirs['standardized']}")
    print(f"过采样数据目录: {output_dirs['oversampled']}")
    print("处理完成")


if __name__ == "__main__":
    main()

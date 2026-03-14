# ===================== 1. 导入核心库 =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # 忽略无用警告

# 设置绘图风格（中文显示+美观）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
sns.set_style("whitegrid")  # 设置绘图背景

# ===================== 2. 读取数据 =====================
# 读取你的训练样本数据（确保csv文件在代码同一文件夹下）
df = pd.read_csv("../data/raw/cs-training.csv")

# ===================== 基础数据概览 =====================
print("="*50)
print("1. 数据集基础信息")
print("="*50)
print(f"数据形状：{df.shape} (行=样本数，列=特征数)")
print("\n列名一览：")
print(df.columns.tolist())
print("\n数据类型&非空值统计：")
print(df.info())
print("\n前5行数据：")
print(df.head())

# ===================== EDA核心1：缺失值分析 =====================
print("\n" + "="*50)
print("2. 缺失值分析（缺失数量+缺失率）")
print("="*50)
# 统计每列缺失数量
missing_count = df.isnull().sum()
# 统计缺失率（百分比）
missing_rate = missing_count / len(df) * 100
# 合并为表格
missing_df = pd.DataFrame({
    "缺失数量": missing_count,
    "缺失率(%)": missing_rate.round(2)
}).sort_values(by="缺失率(%)", ascending=False)

print(missing_df[missing_df["缺失数量"] > 0])  # 只显示有缺失的列

# 可视化：缺失值热力图
plt.figure(figsize=(12, 4))
plt.title("缺失值分布热力图", fontsize=12)
sns.heatmap(df.isnull(), cbar=False, cmap="Reds", yticklabels=False)
plt.tight_layout()
plt.show()

# ===================== EDA核心2：异常值分析 =====================
print("\n" + "="*50)
print("3. 异常值分析（描述性统计）")
print("="*50)
# 数值型特征的最大值、最小值、均值、分位数（快速发现极端值）
print(df.describe().round(2))

# 可视化：核心数值特征箱线图（直观识别异常值）
outlier_cols = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "DebtRatio",
    "MonthlyIncome"
]
plt.figure(figsize=(16, 8))
for i, col in enumerate(outlier_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[col], color="skyblue")
    plt.title(f"{col} 箱线图", fontsize=10)
plt.tight_layout()
plt.show()

# ===================== EDA核心3：特征分布分析 =====================
print("\n" + "="*50)
print("4. 特征分布分析（目标变量+核心特征）")
print("="*50)
# 1. 目标变量分布（SeriousDlqin2yrs：是否违约）
target = "SeriousDlqin2yrs"
print(f"目标变量{target}类别分布：")
print(df[target].value_counts())
print(f"正负样本比例：{df[target].value_counts(normalize=True).round(2)}")

plt.figure(figsize=(12, 10))
# 子图1：目标变量分布
plt.subplot(2, 2, 1)
sns.countplot(x=df[target], palette="viridis")
plt.title("目标变量（是否违约）分布", fontsize=12)

# 子图2：年龄分布
plt.subplot(2, 2, 2)
sns.histplot(df["age"], kde=True, bins=20, color="orange")
plt.title("年龄分布", fontsize=12)

# 子图3：债务率分布
plt.subplot(2, 2, 3)
sns.histplot(df["DebtRatio"], kde=True, bins=30, color="green")
plt.title("债务率分布", fontsize=12)

# 子图4：月收入分布
plt.subplot(2, 2, 4)
sns.histplot(df["MonthlyIncome"].dropna(), kde=True, bins=30, color="purple")
plt.title("月收入分布", fontsize=12)

plt.tight_layout()
plt.show()

# ===================== EDA核心4：特征相关性分析 =====================
print("\n" + "="*50)
print("5. 特征相关性分析")
print("="*50)
# 计算皮尔逊相关系数（数值特征之间的线性相关性）
corr = df.corr()
print("相关系数矩阵（前5行5列）：")
print(corr.round(2))

# 可视化：相关性热力图
plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("特征相关性热力图", fontsize=14)
plt.tight_layout()
plt.show()
# ===================== 模块1：导入所需工具库 =====================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示，避免画图乱码
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False


# ===================== 模块2：加载原始数据集 =====================
# 读取训练集数据，将第一列（用户ID）设为行索引，避免多余列
df = pd.read_csv("data/cs-training.csv", index_col=0)

# 输出加载成功基础信息
print("="*60)
print("✅ 数据集加载成功！")
print(f"数据集总行数（用户样本量）：{df.shape[0]} 条")
print(f"数据集总列数（输入特征数+目标标签列）：{df.shape[1]} 列")
print("="*60)


# ===================== 模块3：特征与标签官方含义对照 =====================
print("\n📌 数据集字段官方含义说明（来自Data Dictionary）：")
feature_desc_dict = {
    "SeriousDlqin2yrs": "【目标标签】用户未来2年是否出现90天以上严重逾期（1=逾期，0=正常）",
    "RevolvingUtilizationOfUnsecuredLines": "无担保信贷额度使用率：信用卡/信用贷等无担保循环信贷余额/授信总额，百分比型",
    "age": "借款人年龄，整数型",
    "NumberOfTime30-59DaysPastDueNotWorse": "过去2年，借款人出现30-59天逾期的次数，整数型",
    "DebtRatio": "负债比例：月债务支出+生活成本/月总收入，百分比型",
    "MonthlyIncome": "月收入，连续数值型",
    "NumberOfOpenCreditLinesAndLoans": "未结清信贷总数：未结清贷款（车贷/房贷）+ 信用卡账户数，整数型",
    "NumberOfTimes90DaysLate": "历史90天以上严重逾期次数，整数型",
    "NumberRealEstateLoansOrLines": "不动产信贷数量：房贷/房屋净值贷总数，整数型",
    "NumberOfTime60-89DaysPastDueNotWorse": "过去2年，借款人出现60-89天逾期的次数，整数型",
    "NumberOfDependents": "家庭抚养人数（不含本人），整数型"
}

# 逐行打印字段含义，方便记录
for col_name, desc in feature_desc_dict.items():
    print(f"▪️ {col_name}：{desc}")
print("="*60)


# ===================== 模块4：数据基本信息探查 =====================
print("\n📊 数据类型与非空值统计：")
# 查看每列的数据类型、非空值数量，快速定位缺失值
df.info()
print("="*60)

print("\n📈 数值特征全局统计分布：")
# 查看均值、标准差、最值、分位数等核心统计指标，保留2位小数方便查看
print(df.describe().round(2))
print("="*60)

print("\n⚠️  缺失值详细统计：")
# 统计每列的缺失值数量和占比，为后续预处理做准备
missing_stats = pd.DataFrame({
    "缺失值数量": df.isnull().sum(),
    "缺失比例(%)": (df.isnull().sum() / df.shape[0] * 100).round(2)
})
# 只显示有缺失的列，过滤无缺失的特征
print(missing_stats[missing_stats["缺失值数量"] > 0])
print("="*60)


# ===================== 模块5：标签分布统计（核心：确认数据不平衡程度） =====================
print("\n🎯 目标标签（逾期/正常用户）分布统计：")
# 统计标签绝对数量
label_count = df["SeriousDlqin2yrs"].value_counts()
# 统计标签占比
label_ratio = df["SeriousDlqin2yrs"].value_counts(normalize=True) * 100

# 合并成可视化表格
label_result = pd.DataFrame({
    "样本数量": label_count,
    "占总样本比例(%)": label_ratio.round(2)
})
# 重命名索引，更直观
label_result.index = ["正常用户(0)", "逾期用户(1)"]
print(label_result)
print("="*60)

# 计算数据不平衡比例
imbalance_rate = label_count[0] / label_count[1]
print(f"⚖️  数据不平衡比例（正常样本:逾期样本）：{imbalance_rate:.2f} : 1")
print(f"结论：逾期样本仅占总样本的{label_ratio[1]:.2f}%，属于典型的不平衡二分类数据集，后续建模需针对性优化")
print("="*60)


# ===================== 模块6：标签分布可视化（用于探索报告截图） =====================
# 1. 柱状图：展示两类用户的绝对数量
plt.figure(figsize=(8, 5))
ax = sns.countplot(
    x="SeriousDlqin2yrs",
    data=df,
    hue="SeriousDlqin2yrs",   # 新增：将 x 赋给 hue
    palette=["#27ae60", "#e74c3c"],
    legend=False               # 新增：关闭图例（因为 x 和 hue 相同）
)

# 图表美化与标注
plt.title("信用卡逾期用户与正常用户数量分布", fontsize=14, fontweight="bold")
plt.xlabel("用户类型（0=正常，1=逾期）", fontsize=12)
plt.ylabel("样本数量", fontsize=12)

# 在柱子上标注具体数值
for p in ax.patches:
    height = p.get_height()
    ax.text(
        p.get_x() + p.get_width()/2., height + 1200,
        f"{int(height)}",
        ha="center", fontsize=11
    )

plt.ylim(0, label_count[0] * 1.1)  # 调整y轴范围，避免标注溢出
plt.tight_layout()
# 保存高清图片到本地，可直接插入毕设报告
plt.savefig("标签分布数量柱状图.png", dpi=300, bbox_inches="tight")
plt.show()

# 2. 饼图：展示两类用户的占比
plt.figure(figsize=(6, 6))
plt.pie(
    label_ratio,
    labels=["正常用户", "逾期用户"],
    autopct="%.2f%%",
    colors=["#27ae60", "#e74c3c"],
    startangle=90,
    textprops={"fontsize": 12}
)
plt.title("信用卡逾期用户占比分布", fontsize=14, fontweight="bold")
plt.axis("equal")  # 保证饼图为正圆形
plt.tight_layout()
plt.savefig("标签占比饼图.png", dpi=300, bbox_inches="tight")
plt.show()


# ===================== 模块7：探索报告总结（可直接复制进毕设） =====================
print("\n📋 数据集探索最终总结报告：")
print(f"1. 数据集规模：共{df.shape[0]}条有效用户样本，包含{df.shape[1]-1}个输入特征，1个二分类目标标签，符合毕设建模的数据量要求；")
print(f"2. 数据质量：存在2个含缺失值的特征，分别是MonthlyIncome（月收入，缺失比例{missing_stats.loc['MonthlyIncome','缺失比例(%)']}%）和NumberOfDependents（家庭抚养人数，缺失比例{missing_stats.loc['NumberOfDependents','缺失比例(%)']}%），需在后续预处理中做填充处理；")
print(f"3. 标签分布：数据集存在严重的类别不平衡，正常用户占比{label_ratio[0]:.2f}%，逾期用户仅占{label_ratio[1]:.2f}%，后续建模需针对不平衡问题做优化（如类别权重调整、过采样/欠采样等）；")
print(f"4. 特征类型：10个输入特征均为数值型，无需做类别编码处理，大幅降低了预处理难度，可直接用于DNN模型输入。")
print("="*60)
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, gaussian_kde
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import NMF, PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sns.set(style='whitegrid')
sns.set_palette('coolwarm')


# 转换为科学计数法，并保留两位小数
def format_scientific(num):
    # 将数字转换为科学计数法，并保留两位小数
    s = "{:.2e}".format(num)
    # 分离出系数和指数
    coef, exp = s.split('e')
    # 将指数转换为带有正负号的格式，并加上'x10^'
    exp = f"x10^{exp}"
    # 拼接系数和指数
    return f"{coef[:3]}{exp}"


# --------1--------
# 读取源文件
raw_dataset = pd.read_csv("CW_Data.csv")
# print(rawDataset)

# # 原始dataframe
# raw_df = pd.DataFrame(raw_dataset)
# 仅包含label
raw_programme = pd.DataFrame(raw_dataset.iloc[:, 3])
# 去除index
raw_dataframe = pd.DataFrame(raw_dataset.iloc[:, 1:11])
# 去除index与Programme
no_label_dataframe = raw_dataset.iloc[:, [1, 3, 4, 5, 6, 7, 8, 9, 10]]

# 使用StandardScaler进行zscore标准化
scaler = StandardScaler()
no_label_df_zscore = scaler.fit_transform(no_label_dataframe)

# 将标准化后的数组转换回DataFrame，并为其分配原始列名
no_label_zscore_column_df = pd.DataFrame(no_label_df_zscore, columns=no_label_dataframe.columns)

# 画对原始数据集zscore后的箱型图
plt.figure(figsize=(10, 6))
plt.boxplot(no_label_zscore_column_df, vert=True, widths=0.5, patch_artist=True)  # 注意这里使用.T进行转置
plt.xticks(range(1, len(no_label_zscore_column_df.columns) + 1), no_label_zscore_column_df.columns)
plt.title('Z-Score Standardized Raw Dataset Features -Box Plot')
plt.xlabel('Feature')
plt.ylabel('Scaled Value')
plt.show()

print(no_label_zscore_column_df.describe())

# 画不同专业的数据分布特征箱型图
columns_to_standardize = ['Total', 'MCQ', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']
# raw_dataframe_zscore = scaler.fit_transform(raw_dataframe)
# 仅对选定的数值列进行 fit 和 transform
numeric_data = raw_dataframe[columns_to_standardize]
df_scaled = scaler.fit_transform(numeric_data)
# 将标准化后的数据转换为 DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=columns_to_standardize)
# 将标准化后的数值列与原始 DataFrame 中的其他列合并
# 注意：保留 'Gender' 列和其他非数值列
raw_dataframe_with_scaled_features = raw_dataframe.drop(columns_to_standardize, axis=1).join(df_scaled)
# # 查看合并后的 DataFrame
# print(raw_dataframe_with_scaled_features)
# 使用 melt 函数将数据重塑为长格式
df_melted = raw_dataframe_with_scaled_features.melt(id_vars=['Gender', 'Programme'],
                                                    var_name='Question',
                                                    value_name='Score')
# 设置图形大小
plt.figure(figsize=(12, 6))
# 使用 seaborn 绘制分组箱形图
sns.boxplot(x='Question', y='Score', hue='Programme', data=df_melted, palette='Set2')
# 设置标题和轴标签
plt.title('Z-Score Standardized Raw Dataset Features -Box Plot Grouped by Programme')
plt.xlabel('Question')
plt.ylabel('Score')
plt.legend(title='Programme', bbox_to_anchor=(1, 1), loc='upper left')
# 显示图形
plt.show()

# --------2--------
# 由于所有Programme3的人都是grade3，绝大多数grade3的同学都是Programme3，可以移除grade3这一特征。

# 去除所有成绩相同的行
# 创建一个包含要检查的列的子集
subset = raw_dataframe[['Total', 'MCQ', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
# 查找重复的行
duplicates = subset.duplicated(keep=False)
# 删除重复的行，保留所有其他列
cleaned_dataframe = raw_dataframe[~duplicates]
# 输出清理后的DataFrame，548行
print(cleaned_dataframe, 'cleaned_dataframe')

# 使用Isolation Forest方法去除异常值
from sklearn.ensemble import IsolationForest

# 选择特征列，排除不需要的列（如索引或标签）
X_iso_progress = cleaned_dataframe.drop(['Gender', 'Programme'], axis=1)
# 初始化Isolation Forest模型，设置contamination参数为异常值的预期比例
clf = IsolationForest(contamination='auto', random_state=0)
# 拟合模型并预测异常值
y_pred = clf.fit_predict(X_iso_progress)
# 找出异常值的索引，-1表示异常值
outlier_indices = y_pred == -1
# 去除异常值的行
cleaned_iso_dataframe = cleaned_dataframe.loc[~outlier_indices]
# 显示清理后的DataFrame
print(cleaned_iso_dataframe, 'cleaned_iso_dataframe')
# cleaned_iso_dataframe.to_csv('cleaned_iso_dataframe.csv', index=False)

# 画每个feature的Q-Q图，是否符合正态分布
exam_columns = no_label_zscore_column_df.columns.intersection(['Total', 'MCQ', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
fig, axs = plt.subplots(nrows=1, ncols=len(exam_columns), figsize=(15, 2.5))  # 调整figsize以适应多个图

for i, column in enumerate(exam_columns):
    stats.probplot(no_label_zscore_column_df[column], dist="norm", plot=axs[i])
    # 对每一列执行正态性检验（样本量较小<5000，使用夏皮罗-威尔克检验法）根据p值判断正态性；均低于0.05，不是正态分布
    w, p_value_shapiro = stats.shapiro(no_label_zscore_column_df[column])

    axs[i].set_title(f'Q-Q plot for {column}\np-value = {format_scientific(p_value_shapiro)}')
    axs[i].set_xlabel('Theoretical quantiles')
    axs[i].set_ylabel('Ordered Values')

plt.tight_layout()  # 调整子图之间的间距
plt.show()

# # 计算每个feature的偏度
# # 初始化一个空字典来存储偏度值
# skewness_dict = {}
# # 计算每列的偏度并存储在字典中
# for column in no_label_zscore_column_df.columns:
#     skewness = skew(no_label_zscore_column_df[column])
#     skewness_dict[column] = skewness
# # 将偏度字典转换为DataFrame，以便更容易查看
# skewness_df = pd.DataFrame.from_dict(skewness_dict, orient='index', columns=['Skewness'])
# # 显示偏度DataFrame
# print(skewness_df)


# 由于非正态分布，使用斯皮尔曼相关性参数检验相关性绘制热力图
# plt.figure(figsize=(12, 10))
# sns.heatmap(raw_dataframe.corr(method='spearman'), annot=True, cmap='coolwarm', fmt=".2f", square=True)
# plt.title('Heatmap of Correlation of every feature')
# plt.show()
plt.figure(figsize=(12, 10))
sns.heatmap(cleaned_dataframe.corr(method='spearman'), annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Heatmap of Spearman rank correlation coefficient of feature-cleaned_dataframe')
plt.show()
plt.figure(figsize=(12, 10))
sns.heatmap(cleaned_iso_dataframe.corr(method='spearman'), annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Heatmap of Correlation of every feature-cleaned_iso_dataframe')
plt.show()

# 经过清洗后的数据的分布情况，画不同专业的数据分布特征箱型图
columns_to_standardize = ['Total', 'MCQ', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']
# raw_dataframe_zscore = scaler.fit_transform(raw_dataframe)
# 仅对选定的数值列进行 fit 和 transform
numeric_data_cleaned = cleaned_iso_dataframe[columns_to_standardize]
df_scaled_cleaned = scaler.fit_transform(numeric_data_cleaned)
# 将标准化后的数据转换为 DataFrame
df_scaled_cleaned = pd.DataFrame(df_scaled_cleaned, columns=columns_to_standardize)
# 将标准化后的数值列与原始 DataFrame 中的其他列合并
# 注意：保留 'Gender' 列和其他非数值列
cleaned_iso_dataframe_with_scaled_features = cleaned_iso_dataframe.drop(columns_to_standardize, axis=1).join(
    df_scaled_cleaned)
# # 查看合并后的 DataFrame
# print(raw_dataframe_with_scaled_features)
# 使用 melt 函数将数据重塑为长格式
cleaned_iso_df_melted = cleaned_iso_dataframe_with_scaled_features.melt(id_vars=['Gender', 'Programme'],
                                                                        var_name='Question',
                                                                        value_name='Score')
# 设置图形大小
plt.figure(figsize=(12, 6))
# 使用 seaborn 绘制分组箱形图
sns.boxplot(x='Question', y='Score', hue='Programme', data=cleaned_iso_df_melted, palette='Set2')
# 设置标题和轴标签
plt.title('Z-Score Standardized clean Dataset Features -Box Plot Grouped by Programme')
plt.xlabel('Question')
plt.ylabel('Score')
plt.legend(title='Programme', bbox_to_anchor=(1, 1), loc='upper left')
# 显示图形
plt.show()

# 使用MIC探索Programme和其他feature的相关性
# 见kaggle


# 使用NMF降维并可视化
# 先归一化
scaler_minmax = MinMaxScaler()
minmax_cleaned_iso_data = scaler_minmax.fit_transform(cleaned_iso_dataframe)
# 获取原始的列名
column_names = cleaned_iso_dataframe.columns
# 将标准化后的数据转换为 DataFrame，并设置列名
minmax_cleaned_iso_dataframe = pd.DataFrame(minmax_cleaned_iso_data, columns=column_names)
print(minmax_cleaned_iso_dataframe)

nmf_feature_input = pd.DataFrame(minmax_cleaned_iso_dataframe.iloc[:, 3:])  # Total MCQ Q1 Q2 Q3 Q4 Q5
nmf_label = pd.DataFrame(minmax_cleaned_iso_dataframe.iloc[:, 1])  # programme
# 确保nmf_label只包含一列，即'Programme'
assert nmf_label.shape[1] == 1, "nmf_label should only contain one column: 'Programme'"
nmf2 = NMF(n_components=2, init='random', random_state=0, max_iter=3000)
W2 = nmf2.fit_transform(nmf_feature_input)

# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(W2[:, 0], W2[:, 1], c=nmf_label.iloc[:, 0], cmap='viridis', marker='o', edgecolor='black')
# for i, txt in enumerate(nmf_label.iloc[:, 0]):
#     plt.annotate(txt, (W[i, 0], W[i, 1]))
plt.xlabel('NMF Component 1')
plt.ylabel('NMF Component 2')
plt.title('NMF Dimensionality Reduction to 2D')
plt.colorbar(label='Programme')
plt.show()

# # 绘制密度图
# plt.figure(figsize=(8, 6))
# sns.kdeplot(x=W2[:, 0], y=W2[:, 1], cmap='viridis', bw_adjust=0.5)
# # 设置标签和标题
# plt.xlabel('NMF Component 1')
# plt.ylabel('NMF Component 2')
# plt.title('NMF Dimensionality Reduction to 2D Density Plot')
# # 显示图形
# plt.show()

# # 使用NMF进行降维，设置n_components=3
# nmf3 = NMF(n_components=3, init='random', random_state=0, max_iter=1000)
# W3 = nmf3.fit_transform(nmf_feature_input)
# # 绘制3D散点图
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# # 绘制散点
# sc = ax.scatter(W3[:, 0], W3[:, 1], W3[:, 2], c=nmf_label.iloc[:, 0], cmap='viridis', marker='o', edgecolor='black')
# # 添加颜色条
# plt.colorbar(sc, label='Programme')
# # 设置轴标签
# ax.set_xlabel('NMF Component 1')
# ax.set_ylabel('NMF Component 2')
# ax.set_zlabel('NMF Component 3')
# # 设置图表标题
# ax.set_title('NMF Dimensionality Reduction to 3D')
# # 旋转3D图
# ax.view_init(elev=10., azim=160)  # 设置仰角为30度，方位角为45度
# # 显示图表
# plt.show()


# # 选择合适的feature，进行pca
# zscore_cleaned_data = scaler.fit_transform(cleaned_dataframe)
# # 获取原始的列名
# column_names = cleaned_dataframe.columns
# # 将标准化后的数据转换为 DataFrame，并设置列名
# zscore_cleaned_dataframe = pd.DataFrame(zscore_cleaned_data, columns=column_names)
#
# pca_input = zscore_cleaned_dataframe.iloc[:, 2:]  # 548x 'Grade', 'Total', 'MCQ', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'
# pca_label = cleaned_dataframe.iloc[:, 1]  # 548x 'Programme'
# print(pca_label)
#
# # 选择其中两个维度降维
# # 创建 PCA 模型并拟合数据,建议去掉性别
# pca = PCA(n_components=8)  # 选择最大的两个维度
# principal_components = pca.fit_transform(pca_input)
# # 将主成分转换为 DataFrame
# principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'])
#
# # 绘制散点图
# plt.figure(figsize=(12, 6))
# sns.scatterplot(x='PC2', y='PC4', data=principal_df, hue=pca_label, palette='viridis', alpha=0.8)
# plt.legend(title='Programme', bbox_to_anchor=(1, 1), loc='upper left')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('PCA Scatter Plot')
# plt.show()
#
# explained_variance = pca.explained_variance_ratio_
# print(pca.components_)
# print('explained_variance:', explained_variance)
#
# # 绘制解释方差比
# plt.bar(range(1, len(explained_variance) + 1), explained_variance)
# plt.xlabel('Principal Component')
# plt.ylabel('Explained Variance Ratio')
# plt.show()
#
# # 累计图
# acc = np.array(explained_variance)
# cumulative_data = np.cumsum(acc)
# # 绘制累积图
# plt.plot(cumulative_data)
# plt.ylim(0, 1)
# plt.xlim(0, 1)
# plt.xticks(range(0, 8))
# plt.show()

# # t-sne降维
#
# # X = rawDataset[['Grade','Q2','Q4','Q5','Total']]
# # X = rawDataset[['Grade','Q5','Total']]
# # X = rawDataset[['Gender','Grade','Total','MCQ','Q1','Q2','Q3','Q4','Q5']]
# X = zscore_cleaned_dataframe[['Grade', 'Total', 'MCQ', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
# y = zscore_cleaned_dataframe['Programme']
#
# # 对使用的数据集标准化
# scaler = StandardScaler()
# x_scaled = scaler.fit_transform(X)
# # 应用t-SNE
# tsne = TSNE(n_components=2, random_state=0)
# X_tsne = tsne.fit_transform(x_scaled)
# tsne_df = pd.DataFrame(X_tsne, columns=['Dimension 1', 'Dimension 2'])
# # 假设 y 是与 X_tsne 相对应的标签序列
# # 我们可以将 y 添加到 tsne_df 中作为一个新的列，用于在 seaborn 中设置 hue
# tsne_df['Label'] = y
# # 可视化结果
# plt.figure(figsize=(10, 6))
# # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
# sns.scatterplot(x='Dimension 1', y='Dimension 2', data=tsne_df, hue='Label', palette='viridis', alpha=0.8)
# plt.legend(title='Programme', bbox_to_anchor=(1, 1), loc='upper left')
#
# plt.xlabel('t-SNE feature 1')
# plt.ylabel('t-SNE feature 2')
# plt.title('t-SNE of zscore_cleaned Dataset')
# plt.show()


# # 对t-sne降维后的结果K-means聚类（监督）
# # 创建K-means聚类模型，设置n_clusters为4
#
# kmeans = KMeans(n_clusters=4, random_state=0)
#
# # 对t-SNE降维后的数据进行K-means聚类
#
# clusters = kmeans.fit_predict(X_tsne)
#
# # 将聚类结果添加到DataFrame中
# tsne_df = pd.DataFrame(X_tsne, columns=['Dimension 1', 'Dimension 2'])
# tsne_df['Cluster'] = clusters
#
# # 可视化结果，使用hue参数显示聚类结果，使用style参数显示Programme标签（如果需要的话）
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='Cluster', data=tsne_df, palette='viridis', alpha=0.8)
# # 如果需要显示Programme标签作为参考，可以取消注释以下行
# # sns.scatterplot(x='Dimension 1', y='Dimension 2', style='Programme', data=tsne_df, palette='Set2', alpha=0.5)
# # 设置图例位置
# plt.legend(title='Cluster', bbox_to_anchor=(1, 1), loc='upper left')
# plt.xlabel('t-SNE feature 1')
# plt.ylabel('t-SNE feature 2')
# plt.title('t-SNE with K-means Clustering of zscore_cleaned Dataset')
# plt.show()


# #tsne降到3维再进行nmf
# # 假设你已经有了一个名为 zscore_cleaned_dataframe 的 DataFrame，并且已经进行了必要的预处理
# X = minmax_cleaned_iso_dataframe[['MCQ', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
# y = minmax_cleaned_iso_dataframe['Programme']
#
# # 数据标准化
# x_scaled = scaler.fit_transform(X)
#
# # 应用t-SNE降至三维
# tsne3 = TSNE(n_components=3, random_state=0)
# X_tsne = tsne3.fit_transform(x_scaled)
# # 将t-SNE结果转换成一个非负值矩阵
# X_tsne_nonnegative = X_tsne - np.min(X_tsne)  # 确保所有值都是非负的
#
# # 将t-SNE的结果转换成一个新的DataFrame，以便于操作
# tsne_df = pd.DataFrame(X_tsne_nonnegative, columns=['Dimension 1', 'Dimension 2', 'Dimension 3'])
# tsne_df['Label'] = y
#
# # 此时，你可以选择对三维t-SNE数据进行NMF分解，尽管这通常不是用于可视化的
# # 但如果你想要进行分解以获取数据的某些潜在结构，可以这样做：
# nmf = NMF(n_components=2, init='random', random_state=10, max_iter=20000)  # 假设我们想要找到2个组件
# W = nmf.fit_transform(tsne_df[['Dimension 1', 'Dimension 2', 'Dimension 3']])
# H = nmf.components_
#
# # W 和 H 分别是NMF分解后的系数矩阵和基矩阵
# # 你可以进一步分析W和H以获取数据的潜在结构信息
#
# # 如果你仍然想要可视化t-SNE的结果，你可以选择两个维度进行绘制
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='Dimension 2', y='Dimension 1', data=tsne_df, hue='Label', palette='viridis', alpha=0.8)
# plt.legend(title='Programme', bbox_to_anchor=(1, 1), loc='upper left')
#
# plt.xlabel('t-SNE feature 1')
# plt.ylabel('t-SNE feature 2')
# plt.title('t-SNE of zscore_cleaned Dataset (3D reduced to 2D for visualization)')
# plt.show()



# 对去除3的情况t-sne降维
zscore_cleaned_iso_data = scaler.fit_transform(cleaned_iso_dataframe)
# 获取原始的列名
column_names = cleaned_iso_dataframe.columns
# 将标准化后的数据转换为 DataFrame，并设置列名
zscore_cleaned_iso_dataframe = pd.DataFrame(zscore_cleaned_iso_data, columns=column_names)
X_iso = zscore_cleaned_iso_dataframe[['MCQ', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
y_iso = zscore_cleaned_iso_dataframe['Programme']

# 对使用的数据集标准化
scaler = StandardScaler()
x_iso_scaled = scaler.fit_transform(X_iso)
# 应用t-SNE
tsne = TSNE(n_components=2, random_state=0)
X_iso_tsne = tsne.fit_transform(x_iso_scaled)
tsne_iso_df = pd.DataFrame(X_iso_tsne, columns=['Dimension 1', 'Dimension 2'])
# 假设 y 是与 X_tsne 相对应的标签序列
# 我们可以将 y 添加到 tsne_df 中作为一个新的列，用于在 seaborn 中设置 hue
tsne_iso_df['Label'] = y_iso
# 可视化结果
plt.figure(figsize=(10, 6))
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
sns.scatterplot(x='Dimension 1', y='Dimension 2', data=tsne_iso_df, hue='Label', palette='viridis', alpha=0.8)
plt.legend(title='Programme', bbox_to_anchor=(1, 1), loc='upper left')

plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.title('t-SNE of zscore_cleaned_iso')
plt.show()


# 2d初始化KPCA
kpca = KernelPCA(n_components=2, kernel='poly')

# 将tsne_df的数值列用于KPCA
X_kpca = kpca.fit_transform(tsne_iso_df[['Dimension 1', 'Dimension 2']])

# 将KPCA的输出转换成一个新的DataFrame
kpca_df = pd.DataFrame(X_kpca, columns=['KPCA 1', 'KPCA 2'])

# 将标签添加到新的DataFrame中
kpca_df['Label'] = y_iso

# 绘制散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x='KPCA 1', y='KPCA 2', data=kpca_df, hue='Label', palette='viridis', alpha=0.8)

# 添加图例标题和轴标签
plt.legend(title='Label', bbox_to_anchor=(1, 1), loc='upper left')
plt.xlabel('KPCA Feature 1')
plt.ylabel('KPCA Feature 2')
plt.title('Kernel PCA of t-SNE Output (poly)')

# 显示图形
plt.show()




# 3维
# 应用t-SNE
tsne3 = TSNE(n_components=3, random_state=0)
X_tsne3 = tsne3.fit_transform(x_iso_scaled)
tsne_df = pd.DataFrame(X_tsne3, columns=['Dimension 1', 'Dimension 2', 'Dimension 3'])
# 假设 y 是与 X_tsne 相对应的标签序列
# 我们可以将 y 添加到 tsne_df 中作为一个新的列，用于在 seaborn 中设置 hue
tsne_df['Label'] = y_iso
# 创建三维散点图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图，使用不同的颜色表示不同的标签
sc = ax.scatter(tsne_df['Dimension 1'], tsne_df['Dimension 2'], tsne_df['Dimension 3'], c=tsne_df['Label'],
                cmap='viridis', marker='o', edgecolor='black', alpha=0.8)
# 添加颜色条
plt.colorbar(sc, label='Programme')
# 设置轴标签
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_zlabel('t-SNE Dimension 3')
# 设置图表标题
ax.set_title('3D Visualization of t-SNE')
ax.view_init(elev=10., azim=230)  # 设置仰角为30度，方位角为45度
# 显示图表
plt.show()


# 初始化KPCA
kpca = KernelPCA(n_components=2, kernel='rbf', fit_inverse_transform=True)

# 将tsne_df的数值列用于KPCA
X_kpca = kpca.fit_transform(tsne_df[['Dimension 1', 'Dimension 2', 'Dimension 3']])

# 将KPCA的输出转换成一个新的DataFrame
kpca_df = pd.DataFrame(X_kpca, columns=['KPCA 1', 'KPCA 2'])

# 将标签添加到新的DataFrame中
kpca_df['Label'] = y_iso

# 绘制散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x='KPCA 1', y='KPCA 2', data=kpca_df, hue='Label', palette='viridis', alpha=0.8)

# 添加图例标题和轴标签
plt.legend(title='Label', bbox_to_anchor=(1, 1), loc='upper left')
plt.xlabel('KPCA Feature 1')
plt.ylabel('KPCA Feature 2')
plt.title('Kernel PCA of t-SNE Output (rbf)')

# 显示图形
plt.show()

# # LDA降维-监督学习
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#
# # 假设你已经有了预处理过的数据框 zscore_cleaned_dataframe 和标签 pca_iso_label
# # 选择特征进行LDA
# lda_input = zscore_cleaned_iso_dataframe[['Total', 'MCQ', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']]  # 假设这是你要使用的特征集
# pca_iso_label = cleaned_iso_dataframe.iloc[:, 1]  # 400x 'Programme'
# # 创建LDA模型对象，假设'Programme'是你的目标变量，有k个不同的类别
# lda = LDA(n_components=3 - 1)  # LDA降维最多降到类别数k-1的维数
# # 拟合LDA模型
# lda_transformed = lda.fit_transform(lda_input, pca_iso_label)
# # 将LDA转换后的数据转换为DataFrame
# lda_df = pd.DataFrame(data=lda_transformed, columns=[f'LD{i + 1}' for i in range(lda_transformed.shape[1])])
#
# # 绘制散点图
# # 因为LDA降维后可能只有1到k-1个组件，所以我们需要根据实际的组件数量来绘制散点图
# n_components = lda_df.shape[1]
# # 如果只有两个组件，直接绘制二维散点图
# if n_components == 2:
#     plt.figure(figsize=(12, 6))
#     sns.scatterplot(x=lda_df.iloc[:, 0], y=lda_df.iloc[:, 1], hue=pca_iso_label, palette='viridis', alpha=0.8)
#     plt.legend(title='Programme', bbox_to_anchor=(1, 1), loc='upper left')
#     plt.xlabel('Linear Discriminant 1')
#     plt.ylabel('Linear Discriminant 2')
#     plt.title('LDA Scatter Plot')
#     plt.show()
# # 如果有多个组件，你可能需要绘制多个二维散点图或者选择其他可视化方法
# else:
#     for i in range(n_components):
#         for j in range(i + 1, n_components):
#             plt.figure(figsize=(12, 6))
#             sns.scatterplot(x=lda_df.iloc[:, i], y=lda_df.iloc[:, j], hue=pca_iso_label, palette='viridis', alpha=0.8)
#             plt.legend(title='Programme', bbox_to_anchor=(1, 1), loc='upper left')
#             plt.xlabel(f'Linear Discriminant {i + 1}')
#             plt.ylabel(f'Linear Discriminant {j + 1}')
#             plt.title(f'LDA Scatter Plot LD{i + 1} vs LD{j + 1}')
#             plt.show()
#
# # # 核主成分分析
# # from sklearn.decomposition import KernelPCA
# #
# # # 选择特征进行KPCA
# # kpca_input = zscore_cleaned_iso_dataframe[['Total', 'MCQ', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']]  # 假设这是你要使用的特征集
# # kpca_iso_label = cleaned_iso_dataframe.iloc[:, 1]  # 400x 'Programme'
# #
# # # 创建KPCA模型对象
# # # 选择一个核函数，比如rbf（径向基函数）
# # kpca = KernelPCA(n_components=2, kernel='poly', fit_inverse_transform=True)
# #
# # # 拟合KPCA模型并转换数据
# # kpca_transformed = kpca.fit_transform(kpca_input)
# #
# # # 将KPCA转换后的数据转换为DataFrame
# # kpca_df = pd.DataFrame(data=kpca_transformed, columns=['KPC1', 'KPC2'])
# #
# # # 绘制散点图
# # plt.figure(figsize=(12, 6))
# # sns.scatterplot(x='KPC1', y='KPC2', data=kpca_df, hue=kpca_iso_label, palette='viridis', alpha=0.8)
# # plt.legend(title='Programme', bbox_to_anchor=(1, 1), loc='upper left')
# # plt.xlabel('Kernel Principal Component 1')
# # plt.ylabel('Kernel Principal Component 2')
# # plt.title('KPCA Scatter Plot')
# # plt.show()
#
#
# # # 字典学习
# # from sklearn.decomposition import DictionaryLearning
# #
# # # 假设你已经有了预处理过的数据框 zscore_cleaned_dataframe
# # # 选择特征进行字典学习降维
# # dl_input = zscore_cleaned_iso_dataframe[['Total', 'MCQ', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']]  # 假设这是你要使用的特征集
# # dl_iso_label = cleaned_iso_dataframe.iloc[:, 1]  # 400x 'Programme'
# # # 创建字典学习模型对象
# # # n_components 指定要学习的字典原子的数量，也就是降维后的特征数
# # dl = DictionaryLearning(n_components=2, alpha=1, random_state=0)
# #
# # # 拟合字典学习模型并转换数据
# # dl_transformed = dl.fit_transform(dl_input)
# #
# # # 将字典学习转换后的数据转换为DataFrame
# # dl_df = pd.DataFrame(data=dl_transformed, columns=[f'DL{i + 1}' for i in range(dl_transformed.shape[1])])
# #
# # # 绘制散点图
# # n_components = dl_df.shape[1]
# #
# # # 根据实际的组件数量来绘制散点图
# # if n_components == 2:
# #     plt.figure(figsize=(12, 6))
# #     sns.scatterplot(x=dl_df.iloc[:, 0], y=dl_df.iloc[:, 1], hue=dl_iso_label, palette='viridis', alpha=0.8)
# #     plt.xlabel('Dictionary Component 1')
# #     plt.ylabel('Dictionary Component 2')
# #     plt.title('Dictionary Learning Scatter Plot')
# #     plt.show()
# # else:
# #     for i in range(n_components):
# #         for j in range(i + 1, n_components):
# #             plt.figure(figsize=(12, 6))
# #             sns.scatterplot(x=dl_df.iloc[:, i], y=dl_df.iloc[:, j], hue=dl_iso_label, palette='viridis', alpha=0.8)
# #             plt.xlabel(f'Dictionary Component {i + 1}')
# #             plt.ylabel(f'Dictionary Component {j + 1}')
# #             plt.title(f'Dictionary Learning Scatter Plot DC{i + 1} vs DC{j + 1}')
# #             plt.show()
#
#
# # # FactorAnalysis因子分析
# # from sklearn.decomposition import FactorAnalysis
# #
# # # 假设你已经有了预处理过的数据框 zscore_cleaned_dataframe
# # # 选择特征进行因子分析降维
# # fa_input = zscore_cleaned_dataframe[['Total', 'MCQ', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']]  # 假设这是你要使用的特征集
# # fa_iso_label = cleaned_iso_dataframe.iloc[:, 1]  # 400x 'Programme'
# # # 创建因子分析模型对象
# # # n_components 指定要提取的因子数量，即降维后的特征数
# # fa = FactorAnalysis(n_components=3)
# #
# # # 拟合因子分析模型并转换数据
# # fa_transformed = fa.fit_transform(fa_input)
# #
# # # 将因子分析转换后的数据转换为DataFrame
# # fa_df = pd.DataFrame(data=fa_transformed, columns=[f'FA{i + 1}' for i in range(fa_transformed.shape[1])])
# #
# # # 绘制散点图
# # n_components = fa_df.shape[1]
# #
# # # 根据实际的组件数量来绘制散点图
# # if n_components == 2:
# #     plt.figure(figsize=(12, 6))
# #     sns.scatterplot(x=fa_df.iloc[:, 0], y=fa_df.iloc[:, 1], hue=fa_iso_label, palette='viridis', alpha=0.8)
# #     plt.xlabel('Factor 1')
# #     plt.ylabel('Factor 2')
# #     plt.title('Factor Analysis Scatter Plot')
# #     plt.show()
# # else:
# #     for i in range(n_components):
# #         for j in range(i + 1, n_components):
# #             plt.figure(figsize=(12, 6))
# #             sns.scatterplot(x=fa_df.iloc[:, i], y=fa_df.iloc[:, j], hue=fa_iso_label, palette='viridis', alpha=0.8)
# #             plt.xlabel(f'Factor {i + 1}')
# #             plt.ylabel(f'Factor {j + 1}')
# #             plt.title(f'Factor Analysis Scatter Plot FA{i + 1} vs FA{j + 1}')
# #             plt.show()
#
#
# # FastICA降维
# from sklearn.decomposition import FastICA
#
# # 假设你已经有了预处理过的数据框 data_frame
# # 选择特征进行FastICA降维
# ica_input = zscore_cleaned_dataframe[['MCQ', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']]  # 假设这是你要使用的特征集
# ica_iso_label = cleaned_iso_dataframe.iloc[:, 1]  # 400x 'Programme'
# # 创建FastICA模型对象
# # n_components 指定要提取的独立成分数量，即降维后的特征数
# ica = FastICA(n_components=4)
#
# # 拟合FastICA模型并转换数据
# ica_transformed = ica.fit_transform(ica_input)
# # 将FastICA转换后的数据转换为DataFrame
# ica_df = pd.DataFrame(data=ica_transformed, columns=[f'ICA{i + 1}' for i in range(ica_transformed.shape[1])])
#
# # # 绘制散点图
# # n_components = ica_df.shape[1]
# # # 根据实际的组件数量来绘制散点图
# # if n_components == 2:
# #     plt.figure(figsize=(12, 6))
# #     sns.scatterplot(x=ica_df.iloc[:, 0], y=ica_df.iloc[:, 1], hue=ica_iso_label, palette='viridis', alpha=0.8)
# #     plt.xlabel('ICA 1')
# #     plt.ylabel('ICA 2')
# #     plt.title('FastICA Scatter Plot')
# #     plt.show()
# # else:
# #     for i in range(n_components):
# #         for j in range(i + 1, n_components):
# #             plt.figure(figsize=(12, 6))
# #             sns.scatterplot(x=ica_df.iloc[:, i], y=ica_df.iloc[:, j], hue=ica_iso_label, palette='viridis', alpha=0.8)
# #             plt.xlabel(f'ICA {i + 1}')
# #             plt.ylabel(f'ICA {j + 1}')
# #             plt.title(f'FastICA Scatter Plot ICA{i + 1} vs ICA{j + 1}')
# #             plt.show()
#
#
# # 在ica降维后形成的空间中进行tsne流形分析
# # 进行t-SNE流形分析
# n_components_tsne = 2  # t-SNE通常降维到2或3维以便可视化
# tsne = TSNE(n_components=n_components_tsne, random_state=0)
# tsne_transformed = tsne.fit_transform(ica_transformed)
#
# # 将t-SNE转换后的数据转换为DataFrame
# tsne_df = pd.DataFrame(data=tsne_transformed, columns=['t-SNE 1', 't-SNE 2'])
#
# # 绘制t-SNE结果的散点图
# plt.figure(figsize=(10, 8))
# sns.scatterplot(x='t-SNE 1', y='t-SNE 2', data=tsne_df, hue=ica_iso_label, palette='viridis', alpha=0.8)
# plt.xlabel('t-SNE 1')
# plt.ylabel('t-SNE 2')
# plt.title('t-SNE Visualization after ICA')
# plt.show()
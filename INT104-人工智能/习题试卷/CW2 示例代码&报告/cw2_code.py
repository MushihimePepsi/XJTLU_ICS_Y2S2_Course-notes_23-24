from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')
sns.set_palette('coolwarm')

# 加载数据集
data_no_g3_only_marks = pd.read_csv("cw_data_clear_grade_only_marks.csv")

# 分离特征和目标变量
X = data_no_g3_only_marks.iloc[:, 1:7]  # 特征 Programme
y = data_no_g3_only_marks.iloc[:, 0]  # 目标变量（Programme） MCQ  Q1  Q2  Q3  Q4  Q5

# 确保X的列名是字符串
X.columns = X.columns.astype(str)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建决策树分类器
clf = DecisionTreeClassifier(max_depth=4)

# 训练模型
clf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = clf.predict(X_test)

# 输出模型的准确度
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 获取所有唯一的类标签，并将它们转换为字符串（如果它们不是字符串的话）
class_labels = y.unique().astype(str)

# 可视化决策树
class_names_str = [str(cls) for cls in sorted(y.unique())]  # 将类别转换为字符串
plt.figure(figsize=(25,10))
plot_tree(clf, filled=True, feature_names=X.columns.tolist(), class_names=class_names_str, fontsize=12)
plt.title('Decision Tree showing the splits at each node')
plt.show()





# 使用h2o分类器自动建模
import h2o
h2o.init()
train, test = data_no_g3_only_marks.split_frame(ratios=[0.8])  # 80%用于训练，20%用于测试
from h2o.automl import H2OAutoML

# 设置自动机器学习的参数
aml = H2OAutoML(max_runtime_secs=600, seed=1)

# 训练模型
aml.train(y="target_column_name", training_frame=train)

# 查看所有模型的性能
lb = aml.leaderboard
print(lb)

# 获取最佳模型
best_model = aml.leader




#随机森林
# 假设我们有以下数据，这里我们使用与上面决策树例子相同的数据
# data = {
#     'Programme': [4, 1, 4, 1, 2],
#     'MCQ': [24, 48, 33, 33, 24],
#     'Q1': [0, 8, 8, 6, 6],
#     'Q2': [0, 8, 2, 2, 2],
#     'Q3': [2, 6, 6, 4, 0],
#     'Q4': [0, 8, 0, 8, 0],
#     'Q5': [0, 3, 0, 1, 0]
# }
# df = pd.DataFrame(data)
#
# # 分离特征和目标变量
# X = df.drop('Programme', axis=1)
# y = df['Programme']
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 创建随机森林分类器实例
# # 这里我们设置了n_estimators为100，表示构建100棵树
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
#
# # 使用训练数据拟合随机森林模型
# rf.fit(X_train, y_train)
#
# # 使用测试集进行预测
# y_pred = rf.predict(X_test)
#
# # 计算准确率
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")
#
# # 输出特征重要性
# importances = rf.feature_importances_
# indices = importances.argsort()[::-1]
# print("Feature ranking by importance:")
# for f in range(X.shape[1]):
#     print(f"%d. feature {X.columns[indices[f]]} (%f)" % (f + 1, importances[indices[f]]))

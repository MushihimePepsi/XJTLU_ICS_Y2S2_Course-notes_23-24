from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from sklearn.model_selection import train_test_split

# Start an H2O cluster
h2o.init()

# 加载数据集
data_no_g3 = pd.read_csv("cw_data_clear_grade3.csv")
X = data_no_g3[['Gender', 'Total', 'MCQ', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']]  # 特征 MCQ  Q1  Q2  Q3  Q4  Q5
y = data_no_g3['Programme']  # 目标变量（Programme）
# 确保X的列名是字符串
X.columns = X.columns.astype(str)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Concatenate training features and target variable for H2O
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

# Convert pandas DataFrame to H2O Frame
train_h2o = h2o.H2OFrame(train)
test_h2o = h2o.H2OFrame(test)

# Set the target variable as a factor (for classification)
train_h2o['Programme'] = train_h2o['Programme'].asfactor()
test_h2o['Programme'] = test_h2o['Programme'].asfactor()

# Define the predictor columns (features) and response column (target)
x = train_h2o.columns
y = 'Programme'
x.remove(y)

# Run H2O AutoML
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=train_h2o)

# View the leaderboard
lb = aml.leaderboard
print(lb)

# Get the best model from AutoML
best_model = aml.leader

# Evaluate model performance on the test set
perf = best_model.model_performance(test_h2o)
print(perf)

# Confusion Matrix: Row labels: Actual class; Column labels: Predicted class
# 1    2    4    Error     Rate
# ---  ---  ---  --------  --------
# 52   3    16   0.267606  19 / 71
# 11   5    23   0.871795  34 / 39
# 21   9    35   0.461538  30 / 65
# 84   17   74   0.474286  83 / 175


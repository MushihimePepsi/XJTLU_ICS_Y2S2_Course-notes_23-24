from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedBaggingClassifier

# Assuming data_no_g3 is your DataFrame containing the data
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
data_no_g3 = pd.read_csv("cw_data_clear_grade3.csv")


# Step 2: Prepare data
X = data_no_g3[['Gender', 'Total', 'Q1', 'Q2', 'Q3', 'Q4']]  # Features
y = data_no_g3['Programme']  # Target variable
# 确保X的列名是字符串
X.columns = X.columns.astype(str)
# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=42)

# Step 3: Undersample majority class
undersample = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = undersample.fit_resample(X_train, y_train)

# Step 5: Build and train the Naive Bayes classifier on undersampled data
gnb_undersampled = GaussianNB()
gnb_undersampled.fit(X_resampled, y_resampled)

# Step 6: Evaluate the Naive Bayes classifier on test data
y_pred_undersampled = gnb_undersampled.predict(X_test)
accuracy_undersampled = accuracy_score(y_test, y_pred_undersampled)
recall_undersampled = recall_score(y_test, y_pred_undersampled, average='weighted')
f1_undersampled = f1_score(y_test, y_pred_undersampled, average='weighted')
conf_matrix_undersampled = confusion_matrix(y_test, y_pred_undersampled)
class_report_undersampled = classification_report(y_test, y_pred_undersampled)

print("Naive Bayes Classifier (Undersampled Data) Performance:")
print("Accuracy:", accuracy_undersampled)
print("Recall:", recall_undersampled)
print("F1 Score:", f1_undersampled)
print("Confusion Matrix:\n", conf_matrix_undersampled)
print("Classification Report:\n", class_report_undersampled)

# Step 7: Build and train a Balanced Bagging classifier
bbc = BalancedBaggingClassifier(estimator=GaussianNB(), sampling_strategy='auto', n_estimators=20, random_state=42)
bbc.fit(X_train, y_train)

# Step 8: Evaluate the Balanced Bagging classifier on test data
y_pred_bbc = bbc.predict(X_test)
accuracy_bbc = accuracy_score(y_test, y_pred_bbc)
recall_bbc = recall_score(y_test, y_pred_bbc, average='weighted')
f1_bbc = f1_score(y_test, y_pred_bbc, average='weighted')
conf_matrix_bbc = confusion_matrix(y_test, y_pred_bbc)
class_report_bbc = classification_report(y_test, y_pred_bbc)

print("\nBalanced Bagging Classifier Performance:")
print("Accuracy:", accuracy_bbc)
print("Recall:", recall_bbc)
print("F1 Score:", f1_bbc)
print("Confusion Matrix:\n", conf_matrix_bbc)
print("Classification Report:\n", class_report_bbc)

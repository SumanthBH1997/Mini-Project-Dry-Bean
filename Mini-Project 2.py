import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the CSV file

df = pd.read_csv(r'C:\Users\shane\Downloads\Documents\Dry_Bean.csv')
print(df)

# Finding unique target class

a = df['Class'].unique()
print(a)

# Getting Data Summary

b = df.describe(percentiles=[.25, .5, .75, .995]).T
print(b)

c = df.info()
print(c)

# Checking for Duplicate Values

d = df.duplicated(subset=None, keep='first').sum()
print(d)

# Exploratory Data Analysis ( EDA )

# Count Plot

print(df['Class'].value_counts())
_ = sns.countplot(x='Class', data=df)

Numeric_cols = df.drop(columns=['Class']).columns

# Histogram

fig, ax = plt.subplots(4, 4, figsize=(15, 12))
for variable, subplot in zip(Numeric_cols, ax.flatten()):
    g = sns.histplot(df[variable], bins=30, kde=True, ax=subplot)
    g.lines[0].set_color('crimson')
    g.axvline(x=df[variable].mean(), color='m', label='Mean', linestyle='--', linewidth=2)
plt.tight_layout()

# Box Plot

fig, ax = plt.subplots(8, 2, figsize=(15, 25))

for variable, subplot in zip(Numeric_cols, ax.flatten()):
    sns.boxplot(x=df['Class'], y=df[variable], ax=subplot)
plt.tight_layout()

fig, ax = plt.subplots(4, 4, figsize=(15, 12))

for variable, subplot in zip(Numeric_cols, ax.flatten()):
    sns.boxplot(y=df[variable], ax=subplot)
plt.tight_layout()

# Heat Map

plt.figure(figsize=(12, 12))
sns.heatmap(df.corr("pearson"), vmin=-1, vmax=1, cmap='coolwarm', annot=True, square=True)

e = plt.show()
print(e)

f = df.head()
print(f)

# Label Encoding Target Column

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
df['Class'] = le.fit_transform(df['Class'])

g = df['Class'].unique()
print(g)

_ = sns.countplot(x='Class', data=df)
h = plt.show()
print(h)

X = df.drop('Area', axis=1)
y = df['Area']

# PCA

from sklearn.decomposition import PCA

pca1 = PCA(n_components=7)
pca_fit1 = pca1.fit_transform(X)

# Train Validation Split

from sklearn.model_selection import train_test_split

features = df.drop(columns=['Class']).columns
X_train, X_test, y_train, y_test = train_test_split(pca_fit1, y, test_size=0.05, random_state=42)

# Importing Models

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

lr = LogisticRegression()
lr.fit(X_train, y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

svc = SVC()
svc.fit(X_train, y_train)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

rm = RandomForestClassifier()
rm.fit(X_train, y_train)

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# Predicition on Test Data

y_pred1 = lr.predict(X_test)
y_pred2 = knn.predict(X_test)
y_pred3 = svc.predict(X_test)
y_pred4 = dt.predict(X_test)
y_pred5 = rm.predict(X_test)
y_pred6 = gb.predict(X_test)

# Evaluating Algorithm

from sklearn.metrics import accuracy_score

print("ACC LR", accuracy_score(y_test, y_pred1))
print("ACC KNN", accuracy_score(y_test, y_pred2))
print("ACC SVC", accuracy_score(y_test, y_pred3))
print("ACC DT", accuracy_score(y_test, y_pred4))
print("ACC RM", accuracy_score(y_test, y_pred5))
print("ACC GBC", accuracy_score(y_test, y_pred6))

final_data = pd.DataFrame({'Models': ['LR', 'KNN', 'SVC', 'DT', 'RM', 'GBC'],
                           'ACC': [accuracy_score(y_test, y_pred1) * 100,
                                   accuracy_score(y_test, y_pred2) * 100,
                                   accuracy_score(y_test, y_pred3) * 100,
                                   accuracy_score(y_test, y_pred4) * 100,
                                   accuracy_score(y_test, y_pred5) * 100,
                                   accuracy_score(y_test, y_pred6) * 100]})

print(final_data)

sns.barplot(final_data['Models'], final_data['ACC'])


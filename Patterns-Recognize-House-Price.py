import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv("test.csv")
print(pd.read_csv('train.csv'))
# 分析“SalePrice”
print(train['SalePrice'].describe())
sn.distplot(train['SalePrice'])
plt.show()

# skewness and kurtosis
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())

corrmat = train.corr()
f, ax = plt.subplots(figsize=(20, 9))
sn.heatmap(corrmat, vmax=0.8, square=True)

from sklearn import preprocessing

f_names = ['CentralAir', 'Neighborhood']
for x in f_names:
    label = preprocessing.LabelEncoder()
    train[x] = label.fit_transform(train[x])
corrmat = train.corr()
f, ax = plt.subplots(figsize=(20, 9))
sn.heatmap(corrmat, vmax=0.8, square=True)

k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sn.set(font_scale=1.25)
hm = sn.heatmap(cm, cbar=True, annot=True, \
                square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(30))

sn.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
sn.pairplot(train[cols], size=2.5)
plt.show()


from sklearn import preprocessing
from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 获取数据
cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
x = train[cols].values
y = train['SalePrice'].values
x_scaled = preprocessing.StandardScaler().fit_transform(x)
y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.33, random_state=42)

clfs = {
    'svm': svm.SVR(),
    'RandomForestRegressor': RandomForestRegressor(n_estimators=600),
    'BayesianRidge': linear_model.BayesianRidge()
}
for clf in clfs:
    try:
        clfs[clf].fit(X_train, y_train)
        y_pred = clfs[clf].predict(X_test)
        print(clf + " cost:" + str(np.sum(y_pred - y_test) / len(y_pred)))
    except Exception as e:
        print(clf + " Error:")
        print(str(e))

cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
x = train[cols].values
y = train['SalePrice'].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf = RandomForestRegressor(n_estimators=900)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)

# 保存clf，共下面计算测试集数据使用
# rfr = clf


y_test

sum(abs(y_pred - y_test)) / len(y_pred)

# 之前训练的模型
rfr = clf


test[cols].isnull().sum()
test[cols].isnull().sum()
test['GarageCars'].describe()
test['TotalBsmtSF'].describe()

cols2 = ['OverallQual', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
cars = test['GarageCars'].fillna(1.766118)
bsmt = test['TotalBsmtSF'].fillna(1046.117970)
test_x = pd.concat([test[cols2], cars, bsmt], axis=1)
test_x.isnull().sum()
x = test_x.values
y_te_pred = rfr.predict(x)
print(y_te_pred)

print(y_te_pred.shape)
print(x.shape)
test_x
prediction = pd.DataFrame(y_te_pred, columns=['SalePrice'])
result = pd.concat([test['Id'], prediction], axis=1)

result.columns

# 保存预测结果
result.to_csv('sample_submission.csv', index=False)

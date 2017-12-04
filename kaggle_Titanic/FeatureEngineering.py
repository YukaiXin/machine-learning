from sklearn.ensemble import RandomForestRegressor
import kaggle_Titanic.DataProcess as titanic_data
import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from  sklearn import linear_model
from sklearn import cross_validation

data_train = titanic_data.titanic_train_data_origin
data_test = titanic_data.titanic_test_data_origin



def set_missing_ages(df):


    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]

    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unkown_age = age_df[age_df.Age.isnull()].as_matrix()


    y = known_age[:, 0]
    X = known_age[:, 1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators= 2000, n_jobs= -1)
    rfr.fit(X, y)

    predicteAges = rfr.predict(unkown_age[:, 1::])

    df.loc[(df.Age.isnull()), 'Age'] = predicteAges

    return  df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin']  = "No"
    return  df

data_train , rfr = set_missing_ages(data_train)
data_train =set_Cabin_type(data_train)


dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis =1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)


#归一化
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].reshape(-1,1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].reshape(-1,1), age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].reshape(-1,1), fare_scale_param)


train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

y = train_np[:,0]
X = train_np[:,1:]

clf = linear_model.LogisticRegression(C=1.0, random_state=0,penalty='l1',tol=0.000001)
clf.fit(X, y)
#-------------------------------------------------------------------------------------
data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0




tmp_df = data_test[['Age','Fare','Parch','SibSp','Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()

test_X = null_age[:,1:]
predictedAges = rfr.predict(test_X)
data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

data_test =set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].reshape(-1,1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].reshape(-1,1), fare_scale_param)

test = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})
result.to_csv(r"C:\Users\yuki_cool\Desktop\work\Titanic\lr_predictions.csv", index=False)


#model系数和feature 关联起来
pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})


# cv_clf = linear_model.LogisticRegression(C=1.0, random_state=0,penalty='l1',tol=0.000001)
#
# all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# X = all_data.as_matrix()[:,1:]
# y = all_data.as_matrix()[:,0]
# #交叉验证打分
# # print(cross_validation.cross_val_score(cv_clf, X, y, cv=5))

split_train, split_cv = cross_validation.train_test_split(df,test_size=0.3,random_state=0)
train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

cv_clf = linear_model.LogisticRegression(C=1.0, random_state=0,penalty='l1',tol=0.000001)
cv_clf.fit(train_df.as_matrix()[:,1:],train_df.as_matrix()[:,0])

#对cross validation 数据进行预测

cv_df =split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = cv_clf.predict(cv_df.as_matrix()[:,1:])

origin_data_train = pd.read_csv(titanic_data.TITANIC_TRAIN_FILE_PATH)
#预测不准的
bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:,0]]['PassengerId'].values)]



##特征的评估......




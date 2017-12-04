from sklearn.ensemble import  BaggingRegressor
import  kaggle_Titanic.FeatureEngineering as FE
from  sklearn import linear_model
import pandas as pd
import numpy as np


train_df = FE.df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

#y即Survival结果
y = train_np[:, 0]

#X即特征属性值
X = train_np[:, 1:]

clf = linear_model.LogisticRegression(C=1.0, random_state=0,penalty='l1',tol=0.000001)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

test = FE.df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = bagging_clf.predict(test)
result = pd.DataFrame({'PassengerId':FE.data_test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})
result.to_csv(r"C:\Users\yuki_cool\Desktop\work\Titanic\bagging_predictions.csv", index=False)



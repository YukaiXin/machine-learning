import numpy as np
import sklearn as skl
import pandas as pd
import csv
import matplotlib.pyplot as plt

TITANIC_TRAIN_FILE_PATH = r"C:\Users\yuki_cool\PycharmProjects\titinaic_data\titanic\train.csv"
TITANIC_TEST_FILE_PATH = r"C:\Users\yuki_cool\PycharmProjects\titinaic_data\titanic\test.csv"


titanic_train_data_origin = pd.read_csv(TITANIC_TRAIN_FILE_PATH,dtype={"Age": np.float64},)
titanic_test_data_origin = pd.read_csv(TITANIC_TEST_FILE_PATH,dtype={"Age": np.float64},)


#
# 0fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
# titanic_train_data_origin["Age"].hist(ax=ax[0])
# ax[0].set_title("Hist plot if Age")
# titanic_train_data_origin["Fare"].hist(ax=ax[1])
# ax[1].set_title("Hist plot of Fase")
# plt.show()

#获救与未获救
# fig,ax = plt.subplots(figsize=(15,5))
# titanic_train_data_origin["Survived"].value_counts().plot(kind="bar")
# ax.set_xticklabels(("Not Survived","Survived"), rotation="horizontal")

#pd.crosstab(titanic_train_data_origin["Sex"],titanic_train_data_origin["Survived"]).plot(kind="bar")


# dummy_Pclass = pd.get_dummies(titanic_train_data_origin.Pclass, prefix='Pclass')
# dummy_Sex = pd.get_dummies(titanic_train_data_origin.Sex, prefix='Sex')
# dummy_Embarked = pd.get_dummies(titanic_train_data_origin.Embarked, prefix='Embarked')
# dummy_Parch = pd.get_dummies(titanic_train_data_origin.Parch, prefix='Parch')
# dummy_SibSp = pd.get_dummies(titanic_train_data_origin.SibSp, prefix='SibSp')
# dummy_Age = pd.get_dummies(titanic_train_data_origin.Age, prefix='Age')
# dummy_Cabin = pd.get_dummies(titanic_train_data_origin.Cabin, prefix='Cabin')
#
#
# train_y = titanic_train_data_origin[:623]["Survived"]
# fare = ["Fare"]
# train_x = titanic_train_data_origin[:623][fare].join(dummy_Sex.ix[:, "Sex_male":]).join(dummy_Embarked.ix[:,"Embarked_Q":]).join(dummy_Parch.ix[:,"Parch_1":]).join(dummy_SibSp.ix[:,"SibSp_1":]).join(dummy_Cabin.ix[:,"Cabin_U" :])
# train_x['intercept'] = 1.0
#
#
# test_y = titanic_train_data_origin[623:]["Survived"]
# fare = ["Fare"]
# test_x = titanic_train_data_origin[:623][fare].join(dummy_Sex.ix[:, "Sex_male":]).join(dummy_Embarked.ix[:,"Embarked_Q":]).join(dummy_Parch.ix[:,"Parch_1":]).join(dummy_SibSp.ix[:,"SibSp_1":]).join(dummy_Cabin.ix[:,"Cabin_U" :])
# test_x['intercept'] = 1.0





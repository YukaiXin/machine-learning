import matplotlib.pyplot as plt
import kaggle_Titanic.DataProcess as titanic_data
import matplotlib
import pandas as pd


data_train = titanic_data.titanic_train_data_origin
zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simhei.ttf')

fig = plt.figure(figsize=(19, 12))
fig.set(alpha=0.2,)

plt.subplot2grid((2,4),(0,0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title("获救情况（1:获救）", fontproperties=zhfont1)
plt.ylabel(u"人数", fontproperties=zhfont1)

plt.subplot2grid((2,4),(0,1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.ylabel(u"人数", fontproperties=zhfont1)
plt.title(u"乘客登级分布", fontproperties=zhfont1)

plt.subplot2grid((2,4),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"年龄", fontproperties=zhfont1)
plt.grid(b=True, which='majior', axis='y')
plt.title(u"按年龄看获救分布（1：获救）", fontproperties=zhfont1)

plt.subplot2grid((2,4),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄", fontproperties=zhfont1)
plt.ylabel(u"密度", fontproperties=zhfont1)
plt.title(u"各等级的年龄分布", fontproperties=zhfont1)
plt.legend((u'头等舱',u'二等舱',u'三等舱'), loc='best', prop =zhfont1)

plt.subplot2grid((2,4), (1,3))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各港口登船人数", fontproperties=zhfont1)
plt.ylabel(u"人数", fontproperties=zhfont1)




#看看各乘客等级的获救情况
plt.subplot2grid((2,4), (0,3))
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客等级的获救情况", fontproperties=zhfont1)
plt.xlabel(u"乘客等级", fontproperties=zhfont1)
plt.ylabel(u"人数", fontproperties=zhfont1)



#看看各登录港口的获救情况

fig.set(alpha=0.3)  # 设定图表颜色alpha参数
plt.subplot2grid((2,4), (1,2))
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各登录港口乘客的获救情况")
plt.xlabel(u"登录港口")
plt.ylabel(u"人数")

#看看各性别的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df=pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
df.plot(kind='bar', stacked=True)
plt.title(u"按性别看获救情况")
plt.xlabel(u"性别")
plt.ylabel(u"人数")
plt.show()






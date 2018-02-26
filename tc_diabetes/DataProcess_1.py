
import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame

DIABETES_TRAIN_DATA = r"C:\Users\yuki_cool\MyCodes\machine-learning\tc_diabetes\datas\d_train_20180102_1.csv";

data_train = pd.read_csv(DIABETES_TRAIN_DATA)
print(data_train.columns)
yx_df = data_train[['id','年龄','*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶', '*碱性磷酸酶', '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白',
 '*球蛋白', '白球比例', '甘油三酯', '总胆固醇', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '尿素', '肌酐', '尿酸', '白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积',
 '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数', '血小板平均体积','血小板体积分布宽度', '血小板比积', '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%','血糖']]

print(data_train.info())

def set_YES_NO(df):
    df.loc[(df['乙肝表面抗原'].notnull()), '乙肝表面抗原'] = "YES"
    df.loc[(df['乙肝表面抗原'].isnull()), '乙肝表面抗原'] = "NO"

    df.loc[(df['乙肝表面抗体'].notnull()), '乙肝表面抗体'] = "YES"
    df.loc[(df['乙肝表面抗体'].isnull()), '乙肝表面抗体'] = "NO"

    df.loc[(df['乙肝e抗原'].notnull()), '乙肝e抗原'] = "YES"
    df.loc[(df['乙肝e抗原'].isnull()), '乙肝e抗原'] = "NO"

    df.loc[(df['乙肝e抗体'].notnull()), '乙肝e抗体'] = "YES"
    df.loc[(df['乙肝e抗体'].isnull()), '乙肝e抗体'] = "NO"

    df.loc[(df['乙肝核心抗体'].notnull()), '乙肝核心抗体'] = "YES"
    df.loc[(df['乙肝核心抗体'].isnull()), '乙肝核心抗体'] = "NO"

data_train = set_YES_NO(data_train)


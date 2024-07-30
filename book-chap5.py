import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

exam = pd.read_csv("data/exam.csv")
# head(), tail(), shape, info(), describe()
exam.head(10)
exam.tail()
exam.shape
exam.info()
exam.describe()

exam2 = exam.copy()
exam2 = exam2.rename(columns = {"nclass" : "class"})
exam2

exam2["total"] = exam2["math"] + exam2["english"] + exam2["science"]
exam2.head()

exam2["test"] = np.where(exam2["total"] >= 200, "pass", "fail")
exam2

exam2["test"].value_counts().plot.bar()

# 200 이상 A
# 100 이상 B
# 100 미만 C
exam2["test2"] = np.where(exam2["total"] >= 200, "A", np.where(exam2["total"] >= 100, "B", "C"))
exam2
plt.show()
plt.clf()

exam2["test2"].isin(["A", "C"])


#데이터 전처리 함수
# query()
# df[]
# sort_values()
# groupby()
# assign()
# agg()
# merge()
# concat()

exam = pd.read_csv("data/exam.csv")
exam.query("nclass == 1")
exam[exam["nclass"] == 1]

exam.query("nclass == 1 & math >= 50")
exam.query("math >= 90 | english >= 90")
exam.query("math >= 90 or english >= 90")
exam.query("nclass in [1, 3, 5]")
exam.query("nclass not in [1, 3, 5]")

~exam["nclass"].isin([1, 2])

exam["nclass"] # Series
exam[["nclass"]] # DataFrame
exam[["id", "nclass"]]
exam.drop(columns = "math")

exam.query("nclass == 1")[["math", "english"]]
exam.sort_values(["nclass", "english"], ascending = [True, False])
exam.assign(total = exam["math"] + exam["english"] + exam["science"])
exam.assign(
  total = exam["math"] + exam["english"] + exam["science"],
  mean = (exam["math"] + exam["english"] + exam["science"] / 3)) \
  .sort_values("total")
  

exam.head()

exam2 = pd.read_csv("data/exam.csv")
exam2 = exam2.assign(
  total = lambda x: x["math"] + x["english"] + x["science"],
  mean = lambda x: (x["math"] + x["english"] + x["science"] / 3))
exam2

exam2.agg(mean_math = ("math", "mean"))
exam2.groupby("nclass").agg(mean_math = ("math", "mean"))

import pydataset
pd.set_option('display.max_columns', None)
mpg = pd.read_csv("data/mpg.csv")
mpg
mpg.query('category == "suv"') \
  .assign(total = (mpg['hwy'] + mpg['cty']) / 2) \
  .groupby('manufacturer') \
  .agg(mean_tot = ('total', 'mean')) \
  .sort_values('mean_tot', ascending = False) \
  .head()

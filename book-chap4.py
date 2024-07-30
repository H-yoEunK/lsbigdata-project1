import pandas as pd
import numpy as np

df = pd.DataFrame({
  'Name': ["김지훈", "이유진", "박동현", "김민지"],
  'English': [90, 80, 60, 70],
  'math': [50, 60, 100, 20]
})

df
type(df)
type(df["Name"])

sum(df["English"]) / 4

df = pd.DataFrame({
  '제품': ["사과", "딸기", "수박"],
  '가격': [1800, 1500, 3000],
  '판매량': [24, 38, 13]
})

sum(df["가격"]) / 3
sum(df["판매량"]) / 3

df[["제품"]]
df[["제품", "가격"]]
type(df[["제품"]])
type(df["제품"])
df

import pandas as pd
df_exam = pd.read_excel("data/excel_exam.xlsx")

sum(df_exam["math"]) / 20
sum(df_exam["english"]) / 20
sum(df_exam["science"]) / 20

len(df_exam)
df_exam.shape
df_exam.size

df_exam = pd.read_excel("data/excel_exam.xlsx", sheet_name="Sheet2")

df_exam["total"] = df_exam["math"] + df_exam["english"] + df_exam["science"]
df_exam["mean"] = df_exam["total"]

df_exam[df_exam["math"] > 50]

df_exam[(df_exam["math"] > 50) & (df_exam["english"] > 50)]

np.mean(df_exam["math"])
df_exam[(df_exam["math"] > df_exam["math"].mean()) & (df_exam["english"] < df_exam["english"].mean())]

df_exam[df_exam["nclass"] == 3][["math", "english", "science"]]

a = np.array([4, 2, 5, 3, 6])
a[2]

df_exam
df_exam[0:10:2]
df_exam[7:16]
df_exam.sort_values("math", ascending = False)
df_exam.sort_values(["nclass", "math"], ascending = [True, False])

np.where(a > 3, "Up", "Down")
df_exam["updown"] = np.where(df_exam["math"] > 50, "Up", "Down")
df_exam

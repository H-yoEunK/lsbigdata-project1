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


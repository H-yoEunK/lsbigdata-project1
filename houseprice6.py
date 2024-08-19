import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

house_train = pd.read_csv("data/houseprice/train.csv")
house_test = pd.read_csv("data/houseprice/test.csv")
sub_df = pd.read_csv("data/houseprice/sample_submission.csv")

# 숫자 변수만 모아서 x로 넣기 (ID, SalePrice 빼기)
x = house_train.select_dtypes(include = [int, float])
x = x.iloc[:, 1:-1]
y = house_train["SalePrice"]
x.isna().sum()

# 평균만
x.fillna(x.mean(), inplace = True)

# 최빈값도 섞어 (성능 안 좋아지더라) -------------------------------------------
fill_values = {
  'LotFrontage' : x['LotFrontage'].mean(),
  'MasVnrArea' : x['MasVnrArea'].mode()[0], # 최빈값
  'GarageYrBlt' : x['GarageYrBlt'].mode()[0]
}
x.fillna(value = fill_values, inplace = True)
# ------------------------------------------------------------------------------

model = LinearRegression()

model.fit(x, y)

model.coef_
model.intercept_

test_x = house_test[x.columns]
test_x.isna().sum()
test_x.fillna(test_x.mean(), inplace = True)

pred_y=model.predict(test_x)
pred_y

sub_df["SalePrice"] = pred_y
sub_df

sub_df.to_csv("./data/houseprice/sample_submission9.csv", index=False)


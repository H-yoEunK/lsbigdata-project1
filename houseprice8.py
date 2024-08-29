import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

house_train=pd.read_csv("./data/houseprice/train.csv")
house_test=pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

house_train.isna().sum()
house_test.isna().sum()

# 숫자형 NaN 채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

# 범주형 NaN 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)
house_train[qual_selected].isna().sum()

house_train.shape
house_test.shape
train_n = len(house_train)

df = pd.concat([house_train, house_test], ignore_index=True)

# 통합 df 만들기 + dummy 코딩
# 범주형 열만 가져오기
# df.select_dtypes(include=[object]).columns

df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )

train_df = df.iloc[:train_n,]
test_df = df.iloc[train_n:,]

np.random.seed(42)
val_index = np.random.choice(np.arange(train_n), size=int(train_n * 0.3), replace=False)

valid_df = train_df.loc[val_index] # 30%
train_df = train_df.drop(val_index) # 70%

train_df = train_df.query("GrLivArea <= 4500")

# x = pd.concat([df[["GrLivArea", "GarageArea"]], 
#              neighborhood_dummies], axis=1)

# 이름 기준으로 특정 열만 추리기
# ^ 시작, $ 끝남, | OR, regex = RegularExpression 정규 방정식
# selected_columns = train_df.filter(regex = '^GrLivArea$|^GarageArea$|^Neighborhood_').columns

train_x = train_df.drop("SalePrice", axis = 1)
train_y = train_df["SalePrice"]

valid_x = valid_df.drop("SalePrice", axis = 1)
valid_y = valid_df["SalePrice"]

test_x = test_df.drop("SalePrice", axis = 1)

model = LinearRegression()
model.fit(train_x, train_y)

model.coef_
model.intercept_

# 성능 측정
y_hat = model.predict(valid_x)
np.sqrt(np.mean((valid_y - y_hat)**2))

# test_y 예측
test_y = model.predict(test_x)
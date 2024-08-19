import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# x에 요인을 2개 넣어보자

house_train = pd.read_csv("data/houseprice/train.csv")
house_test = pd.read_csv("data/houseprice/test.csv")
sub_df = pd.read_csv("data/houseprice/sample_submission.csv")

house_train = house_train.query("GrLivArea <= 4500")

x = house_train[["GrLivArea", "GarageArea"]]
y = house_train["SalePrice"]

model = LinearRegression()

model.fit(x, y)

model.coef_
model.intercept_
# 왜 절편은 하나냐? 두 개의 기울기에 대한 절편이 a, b일 때 둘 다 상수니까 합쳐줌

# -------------------------------------------------------------------------------
def my_f(x):
  return 80.53 * x[0] + 138.994 * x[1] - 18404.7804
# model.coef_[0] * x + model.coef_[1] * y + model.intercept_

my_f([house_test["GrLivArea"], house_test["GarageArea"]])

# -------------------------------------------------------------------------------

#pred_y = model.predict(test_x)
test_x = house_test[["GrLivArea", "GarageArea"]]
test_x["GarageArea"].isna().sum()
test_x["GrLivArea"].isna().sum()
test_x=test_x.fillna(house_test["GarageArea"].mean())

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission7.csv", index=False)

# 3차원 그래프 그리는 코드---------------------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 데이터 포인트
ax.scatter(x['GrLivArea'], x['GarageArea'], y, color='blue', label='Data points')

# 회귀 평면
GrLivArea_vals = np.linspace(x['GrLivArea'].min(), x['GrLivArea'].max(), 100)
GarageArea_vals = np.linspace(x['GarageArea'].min(), x['GarageArea'].max(), 100)
GrLivArea_vals, GarageArea_vals = np.meshgrid(GrLivArea_vals, GarageArea_vals)
SalePrice_vals = intercept + slope_grlivarea * GrLivArea_vals + slope_garagearea * GarageArea_vals

ax.plot_surface(GrLivArea_vals, GarageArea_vals, SalePrice_vals, color='red', alpha=0.5)

# 축 라벨
ax.set_xlabel('GrLivArea')
ax.set_ylabel('GarageArea')
ax.set_zlabel('SalePrice')

plt.legend()
plt.show()
plt.clf()











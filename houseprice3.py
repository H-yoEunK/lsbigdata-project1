# 7장 회귀분석의 이해

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# y = 2x + 3
x = np.linspace(-5, 5, 100)
y = 2 * x + 3
plt.axvline(0, color = "black")
plt.axhline(0, color = "black")
plt.plot(k, y, color = "blue")
plt.show()
plt.clf()

x = np.linspace(0, 5, 100)
y  = 80 * x + 5
house_train = pd.read_csv("data/houseprice/train.csv")
my_df = house_train[["BedroomAbvGr", "SalePrice"]]
my_df["SalePrice"] = my_df["SalePrice"] / 1000

plt.scatter(x = my_df["BedroomAbvGr"], y = my_df["SalePrice"])
plt.plot(x, y, color = "Blue")
plt.show()
plt.clf()


# ------------------------------------------------------------------------------

house_test = pd.read_csv("data/houseprice/test.csv")
a = 53
b = 45

sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df["SalePrice"] = (house_test['BedroomAbvGr'] * a + b) * 1000

sub_df.to_csv("./data/houseprice/sample_submission3.csv", index = False)

# 직선 성능 평가

a = 53
b = 45

# y_hat 어떻게 구할까?
y_hat = (a * house_train["BedroomAbvGr"] + b) * 1000

# y는 어디에 있는가?
y = house_train["SalePrice"]

np.sum(np.abs(y - y_hat)) # 절대 거리의 합 = 직선의 성능

# ------------------------------------------------------------------------------

# !pip install scikit-learn

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 예시 데이터 (x와 y 벡터)
x = np.array([1, 3, 2, 1, 5]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array([1, 2, 3, 4, 5])  # y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습 / 자동으로 기울기, 절편 구해줌
model.fit(x, y)

# 회귀 직선의 기울기와 절편
slope = model.coef_[0] # model.coef_: 기울기 a -> 나중에는 기울기가 여러 개 나올 수 있다
intercept = model.intercept_ # 절편 b
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x) # 기울기랑 절편 알려줬고~ 예측 y 값 반환해줌

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

plt.clf()


# -------------------------------------------------- 위에 걸 houseprice에 적용해보자

house_train = pd.read_csv("data/houseprice/train.csv")

# x = np.array(house_train["BedroomAbvGr"]).reshape(-1, 1)도 가능
x = house_train["BedroomAbvGr"].values.reshape(-1, 1)
y = house_train["SalePrice"].values / 1000

model = LinearRegression()

model.fit(x, y)

# 시각화 ------------------------------------------------------------------------

slope = model.coef_[0]
# 16.38101698298879 -> 방 하나 늘어날 때 1600만 원 정도 늘어난다고 결론 도출?????
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

y_pred = model.predict(x)

plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, y_pred, color='red', label='Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

plt.clf()

# -------------------------------------------------------------------------------


# 회귀 직선 구하기

import numpy as np
from scipy.optimize import minimize

def line_perform(par):
    y_hat=(par[0] * house_train["BedroomAbvGr"] + par[1]) * 1000
    y=house_train["SalePrice"]
    return np.sum(np.abs((y-y_hat)))

line_perform([36, 68])

# 초기 추정 값
initial_guess = [0, 0]

# 최솟값 찾기
result = minimize(line_perform, initial_guess)

# 결과 출력
print("최솟값:", result.fun)
print("최솟값을 갖는 x 값:", result.x)



# y = x^2 + 3의 최솟값이 나오는 입력 값 구하기
def my_f(x):
  return x**2 + 3

my_f(3)

initial_guess = [-10] # 찾는 초기 위치를 지정해줌 (-10에서 조금씩 이동... 경사하강법?)

result = minimize(my_f, initial_guess)

print("최솟값:", result.fun)
print("최솟값을 갖는 x 값:", result.x)


# minimize는 x가 list로 들어온다고 생각하는 애
# 입력 값이 두 개면 x라고 해놓고 첫 번째가 x, 두 번째가 y
def my_f2(x):
  return x[0]**2 + x[1]**2 + 3

my_f2([1, 3,])

initial_guess = [-10, 3] # 찾는 초기 위치를 지정해줌 (-10에서 조금씩 이동... 경사하강법?)

result = minimize(my_f2, initial_guess)

print("최솟값:", result.fun) # -1.11906053e-07 -> 0이나 마찬가지
print("최솟값을 갖는 x 값:", result.x)


def my_f3(x):
  return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 4)**2 + 7

my_f3([1, 3, 6])

initial_guess = [-10, 3, 5]

result = minimize(my_f3, initial_guess)

print("최솟값:", result.fun)
print("최솟값을 갖는 x 값:", result.x)



house_test = pd.read_csv("data/houseprice/test.csv")
test_x = np.array(house_test["BedroomAbvGr"]).reshape(-1, 1)

# 위에 돌려놓은 게 있어서 이것만 씀
pred_y = model.predict(test_x)

sub_df["SalePrice"] = pred_y * 1000
# minimize 함수는 fit 안에 들어 있다

sub_df.to_csv("./data/houseprice/sample_submission4.csv")



# ------------------------------------------------------------------------------

# 이상치 제거 전 GrLivArea = 0.29117
# 이상치 제거 후 GrLivArea = 0.28990

house_train = pd.read_csv("data/houseprice/train.csv")
house_test = pd.read_csv("data/houseprice/test.csv")
sub_df = pd.read_csv("data/houseprice/sample_submission.csv")

# 그냥 다 시각화했을 때 오른쪽 하단에 튀는 점이 있음, 이걸 없애면 더 대표하는 직선을 그릴 수 있을 것
# 이상치 탐색 (일단은 시각화 본 걸로 범위를 잡았는데 나중에는 sort에서 마지막 몇 개를 제거하는 방법도...)
house_train = house_train.query("GrLivArea <= 4500")

# x = house_train[["GrLivArea"]] 하면 np.array랑 reshape 안 해도 됨!!
# house_train["GrLivArea"] 이렇게만 하면 Series로 1차원 벡터, 그래서 하나 더 씌워줌??
x = np.array(house_train["GrLivArea"]).reshape(-1, 1)
y = house_train["SalePrice"]

model = LinearRegression()

model.fit(x, y)

test_x = np.array(house_test["GrLivArea"]).reshape(-1, 1)

pred_y = model.predict(test_x)

sub_df["SalePrice"] = pred_y

sub_df.to_csv("./data/houseprice/sample_submission6.csv", index = False)






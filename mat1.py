import numpy as np

# 벡터 * 벡터 (내적)
a = np.arange(1, 4)
b = np.array([3, 6, 9])

a.dot(b)

a = np.array([1, 2, 3, 4]).reshape((2, 2), order='F')
b = np.array([5, 6]).reshape(2, 1)

a.dot(b)
a @ b

a = np.array([1, 2, 1, 0, 2, 3]).reshape((2, 3))
b = np.array([1, 0, -1, 1, 2, 3]).reshape(2, 3)

# 단위 행렬
np.eye(3)
a = np.array([3, 5, 7,
              2, 4, 9,
              3, 1, 0]).reshape(3, 3)

a @ np.eye(3)
np.eye(3) @ a

a.transpose()
b = a[:,0:2]
b.transpose()

# 회귀 분석 데이터 행렬
x = np.array([13, 15,
              12, 14,
              10, 11,
              5, 6]).reshape(4, 2)

# model.predict가 하는 일
vec1 = np.repeat(1, 4).reshape(4, 1)
matX = np.hstack((vec1, x))
beta_vec = np.array([2, 3, 1]).reshape(3, 1)
matX @ beta_vec

y = np.array([20, 19, 20, 12]).reshape(4, 1)

# 성능
(y - matX @ beta_vec).transpose() @ (y - matX @ beta_vec)

a = np.array([1, 5, 3, 4]).reshape(2, 2)
a_inv = (-1/11) * np.array([4, -5, -3, 1]).reshape(2, 2)

a @ a_inv

# 3 by 3 역행렬
a = np.array([-4, -6, 2,
              5, -1, 3,
              -2, 4, -3]).reshape(3, 3)
a_inv = np.linalg.inv(a)
a_inv

np.round(a @ a_inv, 3)

# 역행렬 존재하지 않는 경우 (선형 종속)
b = np.array([1, 2, 3,
              2, 4, 5,
              3, 6, 7]).reshape(3, 3)
b_inv = np.linalg.inv(b) # 에러남
np.linalg.det(b) # 행렬식이 항상 0

# 벡터 형태로 베타 구하기
XtX_inv = np.linalg.inv((matX.transpose() @ matX))
Xty = matX.transpose() @ y
beta_hat = XtX_inv @ Xty
beta_hat

# model.fit으로 베타 구하기
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(matX[:, 1:], y)

model.coef_
model.intercept_

# minimize로 베타 구하기
from scipy.optimize import minimize

def line_perform(beta):
    beta=np.array(beta).reshape(3, 1)
    a = (y - matX @ beta)
    return (a.transpose() @ a)

line_perform([8.55, 5.96, -4.38])

# 초기 추정값
initial_guess = [0, 1, 0]

result = minimize(line_perform, initial_guess)

print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# minimize로 라쏘 베타 구하기
from scipy.optimize import minimize

def line_perform_lasso(beta):
    beta=np.array(beta).reshape(3, 1)
    a = (y - matX @ beta)
    return (a.transpose() @ a) + 3 * np.abs(beta).sum()

# 일반 회귀 분석은 이걸
line_perform_lasso([8.55, 5.96, -4.38])
# 라쏘 분석은 이걸 선호
line_perform_lasso([3.76, 1.36, 0])

# 초기 추정값
initial_guess = [0, 0, 0]

result = minimize(line_perform_lasso, initial_guess)

print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


# minimize로 릿지 베타 구하기
from scipy.optimize import minimize

def line_perform_ridge(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matX @ beta)
    return (a.transpose() @ a) + 3*(beta**2).sum()

line_perform_ridge([8.55,  5.96, -4.38])
line_perform_ridge([3.76,  1.36, 0])

initial_guess = [0, 0, 0]
result = minimize(line_perform_ridge, initial_guess)

print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)
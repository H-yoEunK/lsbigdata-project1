from scipy.stats import uniform
from scipy.stats import bernoulli
from scipy.stats import binom
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm

old_seat = np.arange(1, 29)

np.random.seed(20240729)
new_seat = np.random.choice(old_seat, 28, replace = False)

result = pd.DataFrame(
  {"old_seat" : old_seat,
  "new_seat" : new_seat}
)

result.to_csv("seat.csv")

# y = 2x 그래프 그리기
import matplotlib.pyplot as plt

x = np.linspace(0, 8 ,2)
# y = uniform.pdf(k, loc = 2, scale = 4)
y = 2 * x
plt.scatter(x, y, s = 2)
# plt.plot(x, y, color = "black")
plt.show()
plt.clf()

# y = x^2를 점 3개를 사용해서 그려보시오
x = np.linspace(-8, 8, 100)
y = x ** 2
# plt.scatter(x, y, s = 2)
plt.plot(x, y, color = "red")
plt.xlim(-10, 10)
plt.ylim(0, 40)
# plt.axis('equal') # x, y 축 비율 맞추기 / xlim, ylim과 같이 사용 X
plt.gca().set_aspect('equal', adjustable = 'box')
plt.show()

plt.clf()


# 통계 교재 57p 신뢰구간 구하기 2)
x = np.array([79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8])
x_bar = x.mean()
n = len(x)
sigma = 6
alpha = 0.1
z_alpha_half = norm.ppf(0.95, loc = 0, scale = 1)

right = x_bar + z_alpha_half * sigma / np.sqrt(16)
left =  x_bar - z_alpha_half * sigma / np.sqrt(16)


# 데이터로부터 E[X^2] 구하기
x = norm.rvs(loc = 3, scale = 5, size = 10000)

np.mean(x**2)
sum(x**2) / (len(x) - 1)

# E[(X - X^2) / 2X]
x = norm.rvs(loc = 3, scale = 5, size = 10000)
np.mean((x-x**2)/(2*x))

# X ~ N(3, 5^2) 표본 10만 개 추출해서 sigma^2을 구해보세요
np.random.seed(20240729)
x = norm.rvs(loc = 3, scale = 5, size = 100000)
x_bar = x.mean()
s_2 = sum((x - x_bar) ** 2) / 99999

# 표본분산을 구하는 위의 과정을 해주는 np의 함수
np.var(x, ddof = 1) # n-1로 나눈 값 (표본분산)
np.var(x, ddof = 0) # n으로 나눈 값 (기본) -> np.var(x) 사용하면 안 됨! 1 차이가 나서



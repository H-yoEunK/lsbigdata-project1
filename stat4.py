from scipy.stats import uniform
from scipy.stats import bernoulli
from scipy.stats import binom
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm

uniform.pdf(rvs, loc = 2, scale = 4)
uniform.pdf(x, loc=0, scale=1)
uniform.cdf(x, loc=0, scale=1)
uniform.ppf(q, loc=0, scale=1)



uniform.rvs(loc=2, scale=4, size=1)
k = np.linspace(0, 8, 100)
y = uniform.pdf(k, loc=2, scale=4)
plt.plot(k, y, color = "black")
plt.show()
plt.clf()

uniform.cdf(3.25, loc = 2, scale = 4)
# 6까지만 0.25의 확률을 가지고 나머지는 0이니까 8.39를 6으로 바꿔도 동일하다!
uniform.cdf(8.39, loc = 2, scale = 4) - uniform.cdf(5, loc = 2, scale = 4)

uniform.ppf(0.93, loc = 2, scale = 4)

# 표본 20개 뽑아서 표본 평균을 계산해 보세요!
x = uniform.rvs(loc=2, scale=4, size=20, random_state = 42)
x.mean()

#-------------------------------------------------------------------------------
# 신뢰구간
# X bar ~ N(mu, sigma^2/n)
# X bar ~ N(4, 1.333/20)
plt.clf()
x_values = np.linspace(3, 5, 100)
pdf_values = norm.pdf(x_values, loc = 4, scale = np.sqrt(1.3333/20))
plt.plot(x_values, pdf_values, color = "red", linewidth = 1)

# 중앙을 기준으로 95%를 구할 때, 왼쪽 거 기준으로 구하려면 0.025를 넣어야 함
a = norm.ppf(0.025, loc = 4, scale = np.sqrt(1.3333/20))
b = norm.ppf(0.975, loc = 4, scale = np.sqrt(1.3333/20))

# 99%
a = norm.ppf(0.005, loc = 4, scale = np.sqrt(1.3333/20))
b = norm.ppf(0.995, loc = 4, scale = np.sqrt(1.3333/20))

# 모평균에 대한 구간 추정 39p
# 정규분포에서 99%를 포함하려면 2.57쯤 되어야 한다
# norm.ppf(0.005, loc = 0, scale = 1) = +-2.57
# 정규분포를 따르는 표본에서 99% 포함 펜스를 치려면 2.57에서 표준편차 만큼을 불린다??
# a = blue_x + 2.57 * np.sqrt(1.333333 / 20)
# b = blue_x - 2.57 * np.sqrt(1.333333 / 20)

# 표본평균(파란 벽돌) 점 찍기
blue_x = uniform.rvs(loc=2, scale=4, size=20).mean()
plt.scatter(blue_x, 0.002, color = 'blue', zorder = 10, s = 10)

# 기댓값 표시
plt.axvline(x = 4, color = 'green', linestyle = '--', linewidth = 2)

plt.show()
#-------------------------------------------------------------------------------

x = uniform.rvs(loc=2, scale=4, size=20*1000, random_state = 42)
x = x.reshape(1000, 20) # 1줄당 20개
blue_x = x.mean(axis = 1)


sns.histplot(blue_x, stat = "density")
plt.show()

# 회색 벽돌을 생성시키는 평균과 분산은 4, 1.333333...
uniform.var(loc = 2, scale = 4)
uniform.expect(loc = 2, scale = 4)

# 파랑 벽돌의 분포를 그렸을 때 평균은 4, 분산은 1.333/20을 따른다
# X bar ~ N(mu, sigma^2/n)
# X bar ~ N(4, 1.333/20)

xmin, xmax = blue_x.min(), blue_x.max()
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc = 4, scale = np.sqrt(1.3333/20))
plt.plot(x_values, pdf_values, color = "red", linewidth = 1)

plt.show()

















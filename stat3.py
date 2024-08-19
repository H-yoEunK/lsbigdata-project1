from scipy.stats import bernoulli
from scipy.stats import binom
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm

# 확률질량함수 (pmf)
# 확률 변수가 갖는 값에 해당하는 확률을 저장하고 있는 함수
# bernoulli.pmf(k, p)

bernoulli.pmf(1, 0.3)
bernoulli.pmf(0, 0.3)

# 이항분포 P(X = k | n, p)
# n: 베르누이 확률 변수 더한 개수
# p: 1이 나올 확률
# binom.pmf(k, n, p)

binom.pmf(0, n=2, p=0.3)
binom.pmf(1, n=2, p=0.3)
binom.pmf(2, n=2, p=0.3)

# X ~ B(n, p)
# list
lst = [binom.pmf(x, n=30, p=0.3) for x in range(31)]

# numpy
binom.pmf(np.arange(31), n=30, p=0.3)

math.factorial(54) / (math.factorial(26) * math.factorial(28))
math.comb(54, 26)

np.cumprod(np.arange(1, 5))[-1]
#fact_54 = np.cumprod(np.arange(1, 55))[-1]

#log(a * b) = log(a) + log(b)
math.log(24)
sum(np.log(np.arange(1, 5)))

math.log(math.factorial(54))
sum(np.log(np.arange(1, 55)))


# log로 큰 수 nCr 계산하기
logf_54 = sum(np.log(np.arange(1, 55)))
logf_26 = sum(np.log(np.arange(1, 27)))
logf_28 = sum(np.log(np.arange(1, 29)))
np.exp(logf_54 - (logf_26 + logf_28))

math.comb(2, 0) * 0.3**0 * (1 - 0.3)**3
math.comb(2, 1) * 0.3**1 * (1 - 0.3)**1
math.comb(2, 2) * 0.3**2 * (1 - 0.3)**0

binom.pmf(0, 2, 0.3)
binom.pmf(1, 2, 0.3)
binom.pmf(2, 2, 0.3)

# X ~ B(n = 10, p = 0.36)
# P(X = 4) = ?
binom.pmf(4, n=10, p=0.36)

# P(X <= 4) = ?
binom.pmf(np.arange(5), n = 10, p = 0.36).sum()

# P(2 < X <= 8) = 8
binom.pmf(np.arange(3, 9), n = 10, p = 0.36).sum()

# X ~ B(30, 0.2)
# 확률변수 X가 4보다 작고, 25보다 크거나 같을 확률을 구하시오
# 1
a = binom.pmf(np.arange(4), n = 30, p = 0.2).sum()
b = binom.pmf(np.arange(26, 31), n = 30, p = 0.2).sum()
a + b

# (강추)2
1 - binom.pmf(np.arange(4, 26), n = 30, p = 0.2).sum()


# rvs (random variates sample)
# 표본 추출 함수
bernoulli.rvs(p = 0.3, size = 1)
bernoulli.rvs(0.3)

# 같다
bernoulli.rvs(0.3) + bernoulli.rvs(0.3)
binom.rvs(n = 2, p = 0.3)

# X ~ B(30, 0.26)
# 표본 30개를 뽑아보세요!
binom.rvs(n = 30, p = 0.26, size = 30)
# n은 총 시행횟수, p는 단일 성공확률, size 생성할 난수의 수

# E[X] = ?
30 * 0.26

# X ~ B(30, 0.26) 시각화
arr = binom.pmf(np.arange(31), 30, 0.26)
plt.bar(np.arange(31), arr)
plt.show()
plt.clf()

sns.barplot(arr)
plt.show()

# 교재 p207 df로 만들어서 막대그래프로 표시하기
x = np.arange(31)
df = pd.DataFrame({"x": x, "prob": arr})
sns.barplot(data = df, x = "x", y = arr)
plt.show()

# cdf: Cumulative Dist. Function
# 누적확률분포 함수
# F_X(x) = P(X <= x)
binom.cdf(4, n=30, p=0.26)
binom.cdf(18, n=30, p=0.26) - binom.cdf(4, n=30, p=0.26)

# P(13 < X < 20) = ?
binom.cdf(19, n=30, p=0.26) - binom.cdf(13, n=30, p=0.26)

# 
plt.clf()
x_1 = binom.rvs(n=30, p=0.26, size=10)
# 성공 확률 0.26의 확률로 30번 했을 때 총합을 10개 생성

x = np.arange(31)
prob_x = binom.pmf(x, n=30, p=0.26)
# x는 전체 시행 가운데 성공의 횟수(구하는 확률 값), n은 전체 시행 횟수, p는 독립 시행의 성공 확률
# prob_x에는 30번 던져서 0번 성공, 1번 성공, ... 30번 성공까지의 확률이 들어감
sns.barplot(prob_x)

plt.scatter(x_1, np.repeat(0.002, 10), color = 'red', zorder = 100, s = 5)
plt.axvline(x = 7.8, color = 'lightgreen', linestyle = '--', linewidth = 2)
plt.show()

binom.ppf(0.5, n = 30, p = 0.26) # 8까지 다 더하면 X = 0.5가 된다는 뜻
binom.cdf(8, n = 30, p = 0.26) # 7 값이 0.463이라서 0.5 넘는 건 8부터다

# 정규분포
1 / np.sqrt(2 * math.pi)
norm.pdf(0, loc = 0, scale = 1)
norm.pdf(5, loc = 3, scale = 4)

# 정규분포 pdf 그리기
k = np.linspace(-5, 5, 100)
y = norm.pdf(k, 0, 1)

plt.clf()
# plt.scatter(k, y, color = 'red', s= 1)
plt.plot(k, y, color = 'black')
plt.show()

# μ(loc): 평균, 분포의 중심을 결정하는 모수
# sigma(scale): 분포의 퍼짐 결정하는 모수
k = np.linspace(-5, 5, 100)
y = norm.pdf(k, 0, 1)
y2 = norm.pdf(k, 0, 2)
y3 = norm.pdf(k, 0, 0.5)

plt.clf()
plt.plot(k, y, color = 'black')
plt.plot(k, y2, color = 'red')
plt.plot(k, y3, color = 'blue')
plt.show()

norm.cdf(0, loc = 0, scale = 1)
norm.cdf(10, loc = 0, scale = 1)

# P(-2 < X < 0.54)
norm.cdf(0.54, loc = 0, scale = 1) - norm.cdf(-2, loc = 0, scale = 1)

# P(X < 1 or X > 3)
1 - (norm.cdf(3, loc = 0, scale = 1) - norm.cdf(1, loc = 0, scale = 1))

# X ~ N(3, 5^2)
# P(3 < X < 5) = ?
norm.cdf(5, loc = 3, scale = 5) - norm.cdf(3, loc = 3, scale = 5)
# 값을 뽑았을 때 3이랑 5 사이에 있을 확률이 15%이다
# 위 확률변수에서 표본 1000개 뽑아보자
x = norm.rvs(loc = 3, scale = 5, size = 1000)
sum((x > 3) & (x < 5)) / 1000

# 평균: 0, 표준편차: 1
# 표본 1000개 뽑아서 0보다 작은 비율 확인
x = norm.rvs(loc = 0, scale = 1, size = 1000)
sum(x < 0) / 1000
(x < 0).mean()

x = norm.rvs(loc = 3, scale = 2, size = 1000)
x
sns.histplot(x, stat = "density")
# sns.histplot(x) 이렇게 하면 x의 빈도가 y로 들어간다
# scale을 맞추기 위해 stat = "density"로 해주면 pdf랑 그렸을 때 딱 맞춰짐 (0 ~ 1)

xmin, xmax = x.min(), x.max()
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc = 3, scale = 2)
plt.plot(x_values, pdf_values, color = "red", linewidth = 1)

plt.show()
plt.clf()

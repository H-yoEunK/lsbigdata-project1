from scipy.stats import norm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x = norm.ppf(0.25, loc = 3, scale = 7)
z = norm.ppf(0.25, loc = 0, scale = 1)
x
3 + z * 7

norm.cdf(5, loc = 3, scale = 7)
norm.cdf(2/7, loc = 0, scale = 7)

norm.ppf(0.975, loc = 0, scale = 1)

# 표본정규분포, 표본 10000개, 히스토그램 -> pdf 겹쳐서 그리기
z = norm.rvs(loc = 0, scale = 1, size = 1000)
z
sns.histplot(z, stat = "density", color = "grey")
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc = 0, scale =1)
plt.plot(z_values, pdf_values, color = "red", linewidth = 2)
plt.show()
plt.clf()

# 이 표본정규분포를 잘 요리해서 X ~ N(3, 2) 꼴로 만들고 싶음
# np.sqrt(2) * z + 3을 해야 함

x = z * np.sqrt(2) + 3
sns.histplot(z, stat = "density", color = "grey")
sns.histplot(x, stat = "density", color = "green")

zmin, zmax = (z.min(), x.min())
z_values = np.linspace(zmin, zmax, 500)
pdf_values = norm.ppf(z_values, loc = 0, scale = 1)
pdf_values2 = norm.ppf(z_values, loc = 3, scale = np.sqrt(2))
plt.plot(z_values, pdf_values, color = "red", linewidth = 2)
plt.plot(z_values, pdf_values2, color = "red", linewidth = 2)

plt.show()
plt.clf()

# X ~ N(5, 3^2)을 따를 때
# Z = (X - 5) / 3이 표준정규분포를 따를까?
x = norm.rvs(loc = 5, scale = 3, size = 1000)
z = (x - 5) / 3
sns.histplot(z, stat = "density", color = "grey")
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc = 0, scale =1)
plt.plot(z_values, pdf_values, color = "red", linewidth = 2)
plt.show()
plt.clf()
# x를 표준화한 z에 대해 막대그래프를 그리고 표준정규분포의 pdf로 plot으로 그렸더니
# 막대그래프와 빨간 선이 겹치네? 표준정규분포가 됐구나 -> 표준화

# 문제
# X 표본을 10개를 뽑아서 표본 분산 값 계산
# X 표본 1000개 뽑음
# 위에서 계산한 s^2으로 모표준편차(sigma^2)을 대체해 표준화 진행
# z의 히스토그램 그리기 = 표준정규분포 pdf 같은지 확인
x = norm.rvs(loc = 5, scale = 3, size = 10)
s = np.std(x, ddof = 1)

x = norm.rvs(loc = 5, scale = 3, size = 1000)
z = (x - 5) / s
sns.histplot(z, stat = "density", color = "grey")
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc = 0, scale =1)
plt.plot(z_values, pdf_values, color = "red", linewidth = 2)
plt.show()
plt.clf()

# 위에서, 표본의 개수가 1000개면 모분산이랑 흡사한데 10개는 잘 안 맞음
# 이렇게 표본의 개수가 적을 때 원래처럼 모분포랑 비슷하게 그리기 위해 ~~~~ t 분포 등장
# 표본의 기준은 상황에 따라 다르긴 한데 30개 이상 넘어가면 어느 정도 되는 듯

# t 분포에 대해서 알아보자
# X ~ t(df)
# 종 모양, 대칭, 중심 0
# 모수 df: 자유도라고 부름 - 분산에 영향을 줌
# df이 작으면 분산 커짐
# df이 무한대로 가면 표준정규분포가 된다
from scipy.stats import t

# t.pdf, t.ppf, t.cdf, t.rvs
# 자유도가 4인 t 분포의 pdf를 그려보세요
t_values = np.linspace(-4, 4, 100)
pdf_values = t.pdf(t_values, df = 4) # df 바꿔가면서 분산 비교
plt.plot(t_values, pdf_values, color = "red", linewidth = 2)

# 표준정규분포와 비교
pdf_values = norm.pdf(t_values, loc = 0, scale = 1)
plt.plot(t_values, pdf_values, color = "black", linewidth = 2)
plt.show()
plt.clf()

# 43p
# X ~ ?(mu, sigma^2)
# X bar ~ N(mu, sigma^2/n)
# X bar ~= t(x_bar, s^2/n) 자유도가 n-1인 t 분포

x = norm.rvs(loc = 15, scale = 3, size = 6, random_state = 24)
x
n = (len(x))
x_bar = x.mean()

# 모분산을 모를 때: 모평균에 대한 95% 신뢰구간을 구해보자!
x_bar + t.ppf(0.975, df = n-1) * np.std(x, ddof = 1) / np.sqrt(n)
x_bar - t.ppf(0.975, df = n-1) * np.std(x, ddof = 1) / np.sqrt()

# 모분산을 알 때: 모평균에 대한 95% 신뢰구간을 구해보자!
x_bar + norm.ppf(0.975, loc = 0, scale = 1) * 3 / np.sqrt(n)
x_bar - norm.ppf(0.975, loc = 0, scale = 1) * 3 / np.sqrt(n)






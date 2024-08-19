import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

# y = 2x + 3 그리기
x = np.linspace(0, 100, 100)
y = 2 * x + 3

# np.random.seed(20240805)
obs_x = np.random.choice(np.arange(100), 20)
epsilon_i = norm.rvs(loc = 0, scale = 20, size = 20)
obs_y = 2 * obs_x + 3 + epsilon_i

sns.lineplot(x = x, y = y, color = "black")
# plt.plot(x, y, color = "black")
plt.scatter(obs_x, obs_y, color = "blue", s = 3)

model = LinearRegression()
obs_x = obs_x.reshape(-1, 1)
model.fit(obs_x, obs_y)

model.coef_
model.intercept_

sns.lineplot(x = x, y = model.coef_ * x + model.intercept_, color = "red")

plt.show()
plt.clf()

# 회귀 직선을 사용해도 될만한 상황이 언제인지를 연구

# !pip install statsmodels
import statsmodels.api as sm

obs_x = sm.add_constant(obs_x)
model = sm.OLS(obs_y, obs_x).fit()
print(model.summary())



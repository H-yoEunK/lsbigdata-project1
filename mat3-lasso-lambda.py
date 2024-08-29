import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

train_df = df.loc[:19]
train_df

for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i

train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y = train_df["y"]

from sklearn.linear_model import Lasso

model= Lasso(alpha=0.1) # alpha = 람다랑 같은 의미
model.fit(train_x, train_y)

model.coef_

valid_df = df.loc[20:]
valid_df

# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택
for i in range(2, 21):
    valid_df[f"x{i}"] = valid_df["x"] ** i

valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]
valid_y = valid_df["y"]

y_hat_train = model.predict(train_x)
y_hat_val = model.predict(valid_x)

sum((train_df["y"] - y_hat_train)**2)
sum((valid_df["y"] - y_hat_val)**2)

# 람다 하나씩 넣어보기 --------------------------------------

val_result = np.repeat(0.0, 100)
tr_result = np.repeat(0.0, 100)

for i in np.arange(0, 100):
    model= Lasso(alpha=i * 0.1)
    model.fit(train_x, train_y)

    y_hat_train = model.predict(train_x)
    y_hat_val = model.predict(valid_x)

    perf_train = sum((train_df["y"] - y_hat_train)**2)
    perf_val = sum((valid_df["y"] - y_hat_val)**2)
    tr_result[i] = perf_train
    val_result[i] = perf_val

import seaborn as sns

df = pd.DataFrame({
    'l': np.arange(0, 10, 0.1),
    'tr': tr_result,
    'val': val_result
})
sns.scatterplot(data=df, x = 'l', y= 'tr')
sns.scatterplot(data=df, x = 'l', y= 'val', color = 'red')
plt.xlim(0, 1)
np.min(val_result) # 이때 alpha는 0.01 -> val에서 가장 성능이 좋은 alpha는 0.01이다

# alpha를 0.01로 선택!
np.argmin(val_result)
np.arange(0, 1, 0.01)[np.argmin(val_result)]

# -----------------------------------------------------------------

model = Lasso(alpha=0.03)
model.fit(train_x, train_y)
model.coef_
model.intercept_
# model.predict(test_x)

k=np.linspace(-4, 4, 800)

k_df = pd.DataFrame({
    "x" : k
})

for i in range(2, 21):
    k_df[f"x{i}"] = k_df["x"] ** i
    
k_df

reg_line = model.predict(k_df)

plt.plot(k_df["x"], reg_line, color="red")
plt.scatter(valid_df["x"], valid_df["y"], color="blue")

# ------------------------------------------------------------
# valid 여러 개 뽑아서 해보기 (내가 짠 거)
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

df = pd.DataFrame({
    "y" : y,
    "x" : x
    })

for i in range(2, 21):
    df[f"x{i}"] = df["x"] ** i

def myf(frac):
    valid_df = df.sample(frac = frac, replace = False)
    train_df = df.drop(valid_df.index)

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
    train_y = train_df["y"]

    valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]
    valid_y = valid_df["y"]

    val_result = np.repeat(0.0, 100)
    tr_result = np.repeat(0.0, 100)

    for i in np.arange(0, 100):
        model= Lasso(alpha = i * 0.01)
        model.fit(train_x, train_y)

        y_hat_train = model.predict(train_x)
        y_hat_val = model.predict(valid_x)

        perf_train = sum((train_df["y"] - y_hat_train)**2)
        perf_val = sum((valid_df["y"] - y_hat_val)**2)
        tr_result[i] = perf_train
        val_result[i] = perf_val
    
    return np.arange(0, 1, 0.01)[np.argmin(val_result)]

result = []
for i in range(5):
    result.append(myf(0.2))

result
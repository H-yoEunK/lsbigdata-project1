import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import Derivative, symbols

beta0 = np.linspace(-20, 20, 400)
beta1 = np.linspace(-20, 20, 400)
beta0, beta1 = np.meshgrid(beta0, beta1)

# 함수 f(x, y)를 계산합니다.
z = (1-(beta0+beta1))**2 + (4-(beta0+2*beta1))**2 + (1.5-(beta0+3*beta1))**2 + (5-(beta0+4*beta1))**2

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(beta0, beta1, z, levels=200)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

plt.scatter(9, 9, color='red', s=10)
beta0 = 9
beta1 = 9
lstep = 0.01
for i in range(1000):
    beta0, beta1 = np.array([beta0, beta1]) - lstep * np.array([8*beta0 + 20*beta1 - 23, 20*beta0 + 60*beta1 - 67])
    plt.scatter(beta0, beta1, color = "red", s=10)


# 축 레이블 및 타이틀 설정
plt.xlabel('beta0')
plt.ylabel('beta1')

# 그래프 표시
plt.show()

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    'x': np.array([1, 2, 3, 4]),
    'y': np.array([1, 4, 1.5, 5])
})

model = LinearRegression()
model.fit(df[['x']], df['y'])

model.intercept_
model.coef_

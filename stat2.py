import numpy as np
import matplotlib.pyplot as plt

# 예제 넘파이 배열 생성
data = np.random.rand(10)

# 히스토그램 그리기
plt.clf()
plt.hist(data, bins = 30, alpha = 0.7, color = 'blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# 5개 뽑아서 10000번 반복하고 그 표본평균을 히스토그램으로 쌓아서 경향 파악
plt.clf()
mean = np.random.rand(50000).reshape(-1, 5).mean(axis = 1)
# mean = np.random.rand(10000, 5).mean(axis = 1)

plt.hist(mean, bins = 30, alpha = 30, color = 'red')
plt.title("Histogram")
plt.xlabel("X_Value")
plt.ylabel("Y_Value")
plt.grid(True)
plt.show()

import numpy as np

np.arange(33).sum() / 33 # 0 ~ 32까지 기댓값

x = np.unique((np.arange(33) - 16) **2) # 중복된 값을 제거

sum(x * (2 / 33))


x = np.arange(33)
sum(x**2 * (1 / 33))

# Var(X) = E[X^2] - (E[X])^2
sum(x**2 * (1/33)) - 16**2

# X = 0, 1, 2, 3이고 확률이 1/6, 2/6, 2/6, 1/6일 때 분산 구하기
x = np.arange(4)
pro_x = np.array([1/6, 2/6, 2/6, 1/6])

Ex = sum(x * pro_x) # 기댓값
Exx = sum(x**2 * pro_x)

Exx - Ex**2

sum((x - Ex)**2 * pro_x)

# 0에서부터 98까지의 정수, 1/2500, 2/2500, 3/2500, ... 50/2500, ... 1/2500일 때 분산은?
x = np.arange(99)
# lst = list(range(1, 50)) + list(range(50, 0, -1))
arr = np.concatenate((np.arange(1, 50), np.arange(50, 0, -1)))

pro_x = arr / 2500

Ex = sum(x * pro_x)
Exx = sum(x**2 * pro_x)
Exx - Ex**2


# Y가 0, 2, 4, 6이고 P(y)가 1/6, 2/6, 2/6, 1/6일 때 분산은?
y = np.arange(4) * 2
pro_y = np.array([1/6, 2/6, 2/6, 1/6])

Ey = sum(y * pro_y)
Eyy = sum(y**2 * pro_y)
Eyy - Ey**2


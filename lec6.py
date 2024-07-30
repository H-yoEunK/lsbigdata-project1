import numpy as np

matrix = np.vstack(
  (np.arange(1, 5),
  np.arange(12, 16))
)

matrix

np.zeros(5)
np.zeros([5, 4])

np.arange(1, 7).reshape([2, 3])
np.arange(1, 7).reshape((2, 3))
np.arange(1, 7).reshape((2, -1)) # -1 통해서 크기를 자동으로 결정할 수 있음

# Q. 0에서 99까지 수 중 랜덤하게 50개 숫자를 뽑아서 5 by 10 행렬을 만드시오
r = np.random.randint(0, 100, 50)
r.shape = (5, 10)

np.random.seed(2024)
a = np.random.randint(0, 100, 50).reshape(5, 10)
a

mat_a = np.arange(1, 21).reshape((4, 5), order = "F")

mat_a[2, 3]
mat_a[0:2, 3]
mat_a[1:3, 1:4]

# 행 자리, 열 자리 비어 있는 경우 전체 행, 또는 열 선택
mat_a[3, ]
mat_a[3, :]
mat_a[3, ::2]

# 짝수 행만 선택하려면
mat_b = np.arange(1, 101).reshape((20, -1))
mat_b
mat_b[1::2,]

mat_b[[1, 4, 6, 14], ]

x = np.arange(1, 11).reshape((5, 2)) * 2
x[[True, True, False, False, True], 0]

mat_b[:, 1] # 1차원으로 돌려줌 (벡터)
mat_b[:, 1:2] # 2차원 유지
mat_b[:, [1]] # 2차원 유지

# 필터링
mat_b[mat_b[:, 1] % 7 == 0, ]
mat_b[mat_b[:, 1] > 50, :]

import matplotlib.pyplot as plt

np.random.seed(2024)
img1 = np.random.rand(3, 3)
print("이미지 행렬 img1:\n", img1)

plt.imshow(img1, cmap = 'gray', interpolation = 'nearest')
plt.colorbar()
plt.show()

a = np.random.randint(0, 10, 20).reshape(4, -1)
a / 9

a = np.random.randint(0, 256, 100).reshape(20, -1)
plt.imshow(a, cmap = 'gray', interpolation = 'nearest')
plt.colorbar
plt.show()

import urllib.request
import imageio

img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "jelly.png")

jelly = imageio.imread("jelly.png")
print(type(jelly))
jelly.shape
jelly[:4, :4, 0] # 이미지 첫 4x4 픽셀, 첫 번째 채널

jelly[:, :, 0].shape
jelly[:, :, 0].transpose().shape

plt.imshow(jelly)
plt.imshow(jelly[:, :, 0].transpose())
plt.imshow(jelly[:, :, 1].transpose())
plt.imshow(jelly[:, :, 2].transpose())
plt.axis('off')
plt.show()
plt.clf

mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)

my_array = np.array([mat1, mat2])
my_array.shape

my_array2 = np.array([my_array, my_array])
my_array2.shape

my_array[0, 1, 1:3]
my_array[0, 1, [1, 2]]

mat_x = np.arange(1, 101).reshape((5, 5, 4))
mat_x = np.arange(1, 101).reshape((10, 5, 2))

a = np.array([[1, 2, 3], [4, 5, 6]])
a.sum()
a.sum(axis = 0)
a.sum(axis = 1)

a.mean()
a.mean(axis = 0)
a.mean(axis = 1)

mat_b = np.random.randint(0, 100, 50).reshape((5, -1))

# 가장 큰 수는?
mat_b.max()

# 행별로 가장 큰 수는?
mat_b.max(axis = 1)

# 열별로 가장 큰 수는?
mat_b.max(axis = 0)

a = np.array([1, 3, 2, 5]).reshape((2, 2))
a.cumsum() # 누적

mat_b.cumsum(axis = 0)

mat_b.reshape((2, 5, 5)).flatten()
mat_b.flatten()

d = np.array([35, 22, 34, 10, 299])
d.clip(30, 35)

d = np.array([1, 2, 3, 4, 5])
d.clip(1, 3)

d.tolist()


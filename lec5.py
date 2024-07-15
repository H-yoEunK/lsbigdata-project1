import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = 2.0
a * b
a.shape
b.shape

# 2차원 배열 생성
matrix = np.array([[ 0.0, 0.0, 0.0],
                  [10.0, 10.0, 10.0],
                  [20.0, 20.0, 20.0],
                  [30.0, 30.0, 30.0]])
matrix.shape
# 1차원 배열 생성
vector = np.array([1.0, 2.0, 3.0])
vector.shape
# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)

vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1)
vector.shape
result = matrix + vector
print("브로드캐스팅 결과:\n", result)

# 벡터 슬라이싱 예제, a를 랜덤하게 채움
np.random.seed(42)
a = np.random.randint(1, 21, 10)
print(a)
# 두 번째 값 추출
print(a[1])

a = [0, 1, 2, 3, 4, 5]
a[-2] # 맨 끝에서 두 번째
a[::2] # [0, 2, 4] 처음부터 끝까지, 스텝은 2
a[0:6:2]

# 1에서부터 1000 사이 3의 배수의 합은?
sum(np.arange(3, 1001)[::3])

np.delete(a, [1,3])

np.random.seed(2024)
a = np.random.randint(1, 10000, 300)
a[a < 5000]

!pip install pydataset
import pydataset

df = pydataset.data('mtcars')
df
np_df = np.array(df['mpg'])

# 15 이상 25 이하인 데이터 개수
sum((np_df >= 15) & (np_df <= 25))

# 평균 mpg보다 이상인 자동차 대수는?
sum(np_df >= np.mean(np_df))

# 15보다 작거나 22 이상인 데이터 개수는?
sum((np_df < 15) | (np_df >= 22))

np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
b = np.array(["A", "B", "C", "F", "W"])
a
a[(a>2000) & (a<5000)]
b[(a>2000) & (a<5000)]

model_names = np.array(df.index)
model_names

# mpg가 15 이상 20 이하인 자동차 모델은?
model_names[(np_df >= 15) & (np_df <= 20)]

# 평균 mpg보다 이상인 자동차 모델은?
model_names[np_df >= np.mean(np_df)]

# 평균 mpg보다 미만인 자동차 모델은?
model_names[np_df <= np.mean(np_df)]

a = np.array([1, 5, 7, 8, 10]) # 예시 배열
result = np.where(a < 7)
result

np.random.seed(2024)
a = np.random.randint(1, 26346, 1000)
# 처음으로 22000보다 큰 숫자가 나왔을 때 위치와 그 숫자는?
a[a>22000][0]
a[np.where(a>22000)]
np.where(a >22000)[0][0]

x = np.where(a > 22000)
type(x)

# 50번째로 10000보다 큰 숫자는?
location = np.where(a > 10000)[0][49]
location
a[location]

# 500보다 작은 숫자들 중 가장 마지막으로 나오는 숫자 위치와 그 숫자는?
location = np.where(a < 500)[0][-1]
location
a[location]

a = np.array([20, np.nan, 13, 24, 309])
np.mean(a)
np.nanmean(a)

np.nan_to_num(a, nan =0)

~np.isnan(a)
a_filtered = a[~np.isnan(a)]
a_filtered

str_vec = np.array(["사과", "배", "수박", "참외"])
str_vec
str_vec[[0, 2]]

mix_vec = np.array(["사과", 12, "수박", "참외"], dtype = str)
mix_vec

combined_vec = np.concatenate((str_vec, mix_vec))
combined_vec = np.concatenate([str_vec, mix_vec])
combined_vec

col_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 16)))
col_stacked

row_stacked = np.row_stack((np.arange(1, 5), np.arange(12, 16)))
row_stacked = np.vstack((np.arange(1, 5), np.arange(12, 16)))
row_stacked

# 길이가 다른 벡터
vec1 = np.arange(1, 5)
vec2 = np.arange(12, 18)
vec1 = np.resize(vec1, len(vec2))
vec1

uneven_stacked = np.column_stack((vec1, vec2))
uneven_stacked

# 1
a = np.array([1, 2, 3, 4, 5])
a + 5

# 2
a = np.array([12, 21, 35, 28, 5])
a[0::2]

# 3
a = np.array([1, 22, 93, 64, 54])
a.max()

# 4
a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
np.unique(a)

# 5
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])

x = np.empty(6)
x[[1, 3, 5]] = b
x[[0, 2, 4]] = a
x

# 또는
x[0::2] = a
x[1::2] = b

np.arange(12, 18)


a = [1, 2, 3]
# soft copy
b = a
a[1] = 4

id(a)
id(b)

c = [1, 2, 3]
d = c[:]
d = c.copy()

c[1] = 4
c
d

import math

sqrt_val = math.sqrt(16)
sqrt_val

exp_val = math.exp(5)
exp_val

log_val = math.log(10, 10)
log_val

fact_val = math.factorial(5)
fact_val

sin_val = math.sin(math.radians(90))
sin_val

cos_val = math.cos(math.radians(180))
cos_val

tan_val = math.tan(math.radians(45))
tan_val


def normal_pdf(x, mu, sigma):
  sqrt_two_pi = math.sqrt(2 * math.pi)
  factor = 1 / (sigma * sqrt_two_pi)
  return factor * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

mu = 0
sigma = 1
x = 1
# 확률밀도함수 값 계산
pdf_value = normal_pdf(x, mu, sigma)

def myf(x, y, z):
  return (x**2 + math.sqrt(y) + math.sin(z)) * math.exp(x)

print(myf(2,9,math.pi/2))

def myg(x):
  return math.cos(x) + math.sin(x) * math.exp(x)

print(myg(math.pi))


import pandas as pd
import numpy as np

a = np.array([1, 2, 3, 4, 5]) # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"]) # 문자형 벡터 생성
c = np.array([True, False, True, True]) # 논리형 벡터 생성
print("Numeric Vector:", a)
print("String Vector:", b)
print("Boolean Vector:", c)
type(a[4])


b = np.empty(3)

vec1 = np.arange(10)
vec1 = np.arange(1.3, 97.0)
vec1 = np.arange(0, 2, 0.5)
vec1

l_space1 = np.linspace(0, 1, 5)
l_space1

vec2 = np.arange(0, 100, -1)
print(vec2)

# 배열 [1, 2, 4]의 각 요소를 각각 1, 2, 3번 반복
repeated_each = np.repeat([1, 2, 4], repeats=[1, 2, 3])
print("Repeated each element in [1, 2, 4] two times:", repeated_each)

repeated_whole = np.tile([1, 2, 4], 2)
repeated_whole

vec1 = np.array([1, 2, 3])
vec2 = np.array([3, 4, 5])
vec1 - vec2

#35672 이하 홀수들의 합은?
result = np.arange(1, 35673, 2)
sum(result)

b = np.array([[1,2,3], [4,5,6]])

length = len(b)
shape = b.shape
size = b.size

length
shape
size

a = np.array([1, 2])
b = np.array([1, 2, 3, 4])
a + b

np.tile(a, 2) + b
np.repeat(a, 2) + b

b == 3

# 35672보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 개수는?
a = np.arange(1, 35672) % 7 == 3
np.count_nonzero(a)
# 또는 (True는 1이고 False는 0이니까!)
sum(np.arange(1, 35672) % 7 == 3)


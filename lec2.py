var1 = [1,2,3]
var2 = [4,5,6]
var1 + var2

a = '안녕하세요!'
a

b = 'LS 빅데이터 스쿨!'
b

print(a)

a = 10
b = 3.3
print("a + b", a + b) # 덧셈
print("a - b =", a - b) # 뺄셈
print("a * b =", a * b) # 곱셈
print("a / b =", a / b) # 나눗셈
print("a % b =", a % b) # 나머지
print("a // b =", a // b) # 몫
print("a ** b =", a ** b) # 거듭제곱
a
b
a == b
a != b
a < b
b < a

(2 ** 4 + 12453 // 7) % 8 < (9 ** 7 // 12) *  (36452 % 253)

TRUE = "hi"
a = "True"
b = TRUE
c = true
d = True

a = True
b = False

# and
True * False
True * True
False * False
False * True

# or
a = True
b = False
a or b
min(a+b, 1)

 a = 3
 a += 10
 a

# 복합 대입 연산자 예제
a = 100
a += 10
print("a += 10:", a) # a = a + 10
a -= 20
print("a -= 20:", a) # a = a - 20
a *= 2
print("a *= 2:", a) # a = a * 2
a /= 2
print("a /= 2:", a) # a = a / 2
a %= 14
print("a %= 14:", a) # a = a % 3
a **= 2
print("a **= 2:", a) # a = a ** 2
a //= 2
print("a //= 2:", a) # a = a // 2

str1 = "Hello! "
str2 = str1 * 3
print("Repeat: ", str2);

x = -4
print("Original x:", x)
print("x", ~x)
bin(x)
bin(~x)
~x

pip install pydataset

import pydataset
pydataset.data()
df = pydataset.data("AirPassengers")
df
import pandas
import numpy

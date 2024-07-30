fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "apple", 3.5, True]

empty_list1 = []
empty_list2 = list()

# 초기값을 가진 리스트 생성
numbers = [1, 2, 3, 4, 5]
range_list = list(range(5))

range_list[3] = "LS 빅데이터 스쿨"
range_list

range_list[1] = ["배", "고", "파"]
range_list

range_list[1][2]

# 리스트 내포
# 1. 대괄호로 싸여 있다 => 리스트다
# 2. 넣고 싶은 수식을 x를 사용해서 표현
# 3. for .. in .. 을 사용해서 원소 정보 제공

squares = [x ** 2 for x in range(10)]

list(range(10))

my_squares = [x ** 3 for x in [3, 5, 2, 15]]

# numpy array
import numpy as np
np.array([3, 5, 2, 15])
my_squares = [x ** 3 for x in np.array([3, 5, 2, 15])]

# Pandas Series
import pandas as pd
import matplotlib
exam = pd.read_csv("data/exam.csv")
my_squares = [x ** 3 for x in exam["math"]]

list1 = [1, 2, 3]
list2 = [4, 5, 6]
list1 * 3 + list2 * 5

numbers = [5, 2, 3]
repeated_list = [x for x in numbers for _ in range(3)] # 앞 해석이 먼저, 5가 randge(3) 만큼 3번 반복된다. 그 다음은 2~
repeated_list

# 리스트를 하나 만들어서 for loop 사용해서 2, 4, 6, 8, ... , 20의 수를 채워 보세요!
list_ = [ x for x in range(2, 21, 2)]

lst = []
for i in range(2, 21, 2):
  lst.append(i)

mylist_b = [2, 4, 6, 8, 80, 10, 12, 24, 35, 23, 23, 20]
mylist = [0] * 10

for i in range(10):
  mylist[i] = mylist_b[i]

mylist

# 퀴즈: mylist_b의 홀수번째 위치에 있는 숫자들만 mylist에 가져오기
mylist_b = [2, 4, 6, 8, 80, 10, 12, 24, 35, 23, 23, 20]
mylist = [0] * 10

for i in range(5):
  mylist[i] = mylist_b[i * 2]

# 리스트 컴프리헨션으로 바꾸는 방법
# 바깥은 무조건 대괄호로 묶어줌: 리스트 반환하기 위해서
# for 루프의 :는 생략한다
# 실행 부분을 먼저 써준다
# 결과를 받는 부분 제외시킴

mylist = []
[i * 2 for i in range(1, 11)]

for i in [0, 1, 2]:
  for j in [0, 1]:
    print(i, ",", j)

# 리스트 컴프리헨션 변환
[i for i in range(3) for j in range(2)]

[j for j in range(3) for j in range(2)] # 뒤에 있는 반복문이 j를 0, 1로 바꿔주기 때문에 앞에 있는 j가 숫자에 영향 X, 앞 먼저 실행이 맞음

# 원소 체크
fruits = ["apple", "banana", "cherry"]
"banana" in fruits

for x in fruits:
  x == "banana"

fruits.index("banana")

import numpy as np
fruits = np.array(fruits)
np.where(fruits == "banana")[0][0]
int(np.where(fruits == "banana")[0][0])

# 넘파이 배열 생성
fruits = np.array(["apple", "banana", "cherry", "apple", "pineapple"])
# 제거할 항목 리스트
items_to_remove = np.array(["banana", "apple"])
# 불리언 마스크 생성
mask = ~np.isin(fruits, items_to_remove)
mask = ~np.isin(fruits, ["banana", "apple"]) # 가능
# 불리언 마스크를 사용하여 항목 제거
filtered_fruits = fruits[mask]
print("remove() 후 배열:", filtered_fruits)


# 여러 항목을 제거하는 방법
fruits = ["apple", "banana", "cherry", "apple", "banana"]
# 반복문을 사용하여 항목 제거
for item in ["banana", "apple"]: # while 조건문: True일 때 실행
  while item in fruits:
    fruits.remove(item)

print("remove() 후 리스트:", fruits)

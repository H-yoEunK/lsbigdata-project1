x = 15
print(x, "는 ", type(x), "형식입니다.", sep='')

# 여러 줄 문자열
ml_str = """This is
a multi-line
string"""
print(ml_str, type(ml_str))

tmp = "this \n is \n multi"
print(tmp)

fruit = ["apple", "banana", "cherry"]
type(fruit)

numbers = [1, 2, 3, 4]
type(numbers)

mixed_list = [1, "Hello", [1, 2, 3]]
type(mixed_list)

a = (10, 20, 30) # a = 10, 20, 30과 동일
b = (42)
type(b)
b = (42,)
type(b)

a = [10, 20, 30]
b = (10, 20, 30)
a[1]
b[1]
a[1] = 100
b[1] = 100

a_tp = (10, 20, 30, 40, 50)
a_tp[3:] # 해당 인덱스 이상
a_tp[:3] # 해당 인덱스 미만
a_tp[1:3] # 1 이상 3 미만

a_lst = [10,20,30,40,50]
a_lst[3:]
a_lst[:3]
a_lst[1:3]

# 사용자 정의 함수
def min_max(numbers):
  return min(numbers), max(numbers)

# result = min_max([1, 2, 3, 4, 5])
result = min_max((1, 2, 3, 4, 5))
print("Minimum and maximum", result)

person = {
  'name' : 'John',
  'age' : 30,
  'city' : ['New York', 'Korea', 'Paris']
}

print("Person:", person)

person.get('city')[1]

# 집합
fruits = {'apple', 'banana', 'cherry', 'apple'}
empty_set = set()
print(empty_set)

empty_set.add('apple')

is_active = True
is_greater = 10 > 5 # True 반환
is_equal = (10 == 5) # False 반환
print("Is active:", is_active)
print("Is 10 greater than 5?:", is_greater)
print("Is 10 equal to 5?:", is_equal)

set_example = {'a', 'b', 'c'}
# 딕셔너리로 변환 시, 일반적으로 집합 요소를 키 또는 값으로 사용
dict_from_set = {key: True for key in set_example}
print("Dictionary from set:", dict_from_set)

# 문자열을 논리형으로 변환
str_true = "True"
str_false = "False"
bool_from_str_true = bool(str_true) # True
bool_from_str_false = bool(str_false) # True, 비어있지 않으면 무조건 참
print("'True'는 논리형으로 바꾸면:", bool_from_str_true)
print("'False'는 논리형으로 바꾸면:", bool_from_str_false)

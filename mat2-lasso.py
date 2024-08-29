import numpy as np

# 회귀분석 데이터행렬
x=np.array([13, 15,
           12, 14,
           10, 11,
           5, 6]).reshape(4, 2)
x
vec1=np.repeat(1, 4).reshape(4, 1)
matX=np.hstack((vec1, x))
y=np.array([20, 19, 20, 12]).reshape(4, 1)
matX

# minimize로 라쏘 베타 구하기
from scipy.optimize import minimize

def line_perform_lasso(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matX @ beta)
    return (a.transpose() @ a) + 3 * np.abs(beta[1:]).sum()

line_perform_lasso([8.55,  5.96, -4.38])
line_perform_lasso([3.76,  1.36, 0])
initial_guess = [0, 0, 0]

result = minimize(line_perform_lasso, initial_guess)

print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# [8.14, 0.96, 0]
# 예측 식: y_hat = 8.14, 0.96 * X1 + 0 * X2
# X2의 특징은 해당 선형 회귀 분석에서 가중치를 가지지 않는다... 필요 없는 특성

# [8.55, 5.96, -4.38] 람다 0 = 선형 회귀 분석
# [8.14, 0.96, 0] 람다 3
# [17.74, 0, 0] 람다 500 -> 조금 건드려서 패널티 키울바에 그냥 0 준다 (underfitting)

# 람다 값에 따라 변수 선택된다
# X 변수가 추가되면 train X에서는 성능 항상 좋아짐
# X 변수가 추가되면 valid X에서는 좋아졌다가 나빠짐 (오버피팅)
# 이 의미가 무엇이냐
# X 변수가 추가되는 건 몸무게를 유추하는 데에 키, 머리 둘레, 허리 길이 ... 등의 변수가 늘어나는 건데
# Overfitting Code에서 x^2, x^3 ... 했던 것처럼
# X 변수가 늘어나면 차수가 늘어난다고 볼 수 있다 (X^2도 새로운 X2?)
# 그럼 X가 12개일 때 모델이 복잡해진다 = 차수가 늘어나서 12개의 곡선을 꼭 만들어야 함
# 그러다보면 train 노이즈 하나 하나에 영향을 크게 받으면서 곡선이 구불구불 크게 그려지고, train에 최적화됨 (Overfitting)
# 이 곡선을 가지고 valid에 적용하니 나빠질 수밖에,,,~!
# 어느 순간 X 변수 추가하는 것을 멈춰야 함
# 람다 0부터 시작: 내가 가진 모든 변수를 넣겠다!
# 점점 람다를 증가: 변수가 하나씩 빠지는 효과
# valid X에서 가장 성능이 좋은 람다를 선택!
# 변수가 선택됨을 의미

# (X^T X)^-1
# X의 칼럼 선형 독립이어야 한다 -> 아니면 fit 할 때 에러가 남
# 라쏘나 릿지는 X들이 선형 독립이 아니더라도 inv가 항상 존재해서 베타가 구해짐 (장점)
# 다중공선성은 역행렬이 안 나와서??

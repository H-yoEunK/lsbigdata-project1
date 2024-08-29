import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

# lambda 9.99
# 1.68532

house_train=pd.read_csv("./data/houseprice/train.csv")
house_test=pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

house_train.isna().sum()
house_test.isna().sum()

quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)
house_train[qual_selected].isna().sum()

house_train.shape
house_test.shape

df = df.query("GrLivArea <= 4500")
train_n = len(house_train)
df = pd.concat([house_train, house_test], ignore_index=True)

df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )

train_df = df.iloc[:train_n,]
test_df = df.iloc[train_n:,]

x = train_df.drop("SalePrice", axis = 1)
y = train_df["SalePrice"]


kf = KFold(n_splits=10, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, x, y, cv = kf,
                                     n_jobs=-1, scoring = "neg_mean_squared_error").mean())
    return(score)

alpha_values = np.arange(0, 10, 0.01)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

mean_scores

np.argmin(mean_scores)
np.arange(0, 10, 0.01)[np.argmin(mean_scores)]

quantitative = test_df.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    test_df[col].fillna(test_df[col].mean(), inplace=True)
test_df[quant_selected].isna().sum()

qualitative = test_df.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    test_df[col].fillna("unknown", inplace=True)
test_df[qual_selected].isna().sum()

model= Lasso(alpha=9.99)
model.fit(x, y)


test_df = test_df.drop('SalePrice', axis = 1)
pred_y = model.predict(test_df)

sub_df["SalePrice"] = -pred_y
sub_df

sub_df.to_csv("./data/houseprice/sample_submission11.csv", index=False)

# ------------------------------------------------------------------------
# 다른 조 코드

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error


## 필요한 데이터 불러오기
house_train=pd.read_csv("data/train.csv")
house_test=pd.read_csv("data/test.csv")
sub_df=pd.read_csv("data/sample_submission.csv")

# NaN 채우기
house_train.isna().sum()
house_test.isna().sum()

# 각 숫자변수는 평균채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)

house_train[quant_selected].isna().sum()

# 각 변수형변수는 최빈값으로 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna(house_train[col].mode(), inplace=True)

house_train[qual_selected].isna().sum()


train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
df = pd.concat([house_train, house_test], ignore_index=True)

df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include = [object]).columns,
    drop_first=True
    )
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

X = train_df.drop(["SalePrice", "Id"], axis = 1)
y = train_df['SalePrice']

# 교차 검증 설정
kf = KFold(n_splits=20, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X, y, cv = kf,
                                     n_jobs = -1, scoring = "neg_mean_squared_error").mean())
    return(score)


# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(120, 125, 0.01)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df_result = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

df_result

# 결과 시각화
plt.plot(df_result['lambda'], df_result['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기: 120
optimal_alpha = df_result['lambda'][np.argmin(df_result['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# 라쏘 회귀 모델 생성
model = Lasso(alpha = 120)

# 모델 학습
model.fit(X, y) 

test_df=df.iloc[train_n:,]
test_df = test_df.drop(["SalePrice", "Id"], axis = 1)

quantitative = test_df.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    test_df[col].fillna(test_df[col].mean(), inplace=True)

test_df[quant_selected].isna().sum()

# 각 변수형변수는 최빈값 채우기
qualitative = test_df.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    test_df[col].fillna(test_df[col].mode(), inplace=True)

test_df[qual_selected].isna().sum()

pred_y = model.predict(test_df)

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/sample_submission12.csv", index=False)
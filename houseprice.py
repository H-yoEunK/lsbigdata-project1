import pandas as pd
import numpy as np


traindf = pd.read_csv("data/houseprice/train.csv")
mean = traindf["SalePrice"].mean()

subdf = pd.read_csv("data/houseprice/sample_submission.csv")
subdf["SalePrice"] = mean

subdf.to_csv("./data/houseprice/sample_copy.csv", index = False)

# ------------------------------- 조별 활동 (YearBuilt 기준으로 값 예측)

# 연도 범위, 평균 확인
house = pd.read_csv("data/houseprice/train.csv")

# 년도 별 그룹바이, 가격 평균
new = house.groupby('YearBuilt',as_index = False) \
           .agg(new_price = ('SalePrice','mean'))
new

# test 불러오기
test = pd.read_csv('data/houseprice/test.csv')

# test랑 가격 평균 합체
new2 = pd.merge(test, new, how = 'left', on = 'YearBuilt')

pd.isna(new2).sum()

# new price 결측치 전체 평균값넣기
new2 = new2.fillna(new2['new_price'].mean())

# 제출용데이터 불러오기
sub = pd.read_csv('house/sample_submission.csv')
sub2 = sub.copy()
#제출용 데이터에 년도별 그룹합치기
sub2['SalePrice'] = new2['new_price']
sub2.to_csv('./house/sub2.csv', index = False)

# --------------------------------------------------- 강사님과 함께 풀이

house_train = pd.read_csv("data/houseprice/train.csv")
house_train = house_train[["Id","YearBuilt", "SalePrice"]]
house_train.info()

house_mean = house_train.groupby("YearBuilt", as_index = False) \
            .agg(mean_year = ("SalePrice", "mean"))
            
house_test = pd.read_csv("data/houseprice/test.csv")
house_test = house_test[["Id", "YearBuilt"]]

house_test = pd.merge(house_test, house_mean, how = "left", on = "YearBuilt")

house_test = house_test.rename(columns = {"mean_year" : "SalePrice"})

sum(house_test["SalePrice"].isna())

# 결측치 있는 행 확인
house_test.loc[house_test["SalePrice"].isna()]
house_mean = house_train["SalePrice"].mean()
house_test["SalePrice"] = house_test["SalePrice"].fillna(house_mean)

sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df["SalePrice"] = house_test["SalePrice"]

sub_df.to_csv("./data/houseprice/sample_submission2.csv", index = False)

# -------------------------------------------------------------------------

train = pd.read_csv("data/houseprice/train.csv")
train = train[["Id", "1stFlrSF", "2ndFlrSF", "Neighborhood", "SalePrice"]]
train = train.assign(FlrSF = train["1stFlrSF"] + train["2ndFlrSF"])
            
train['size'] = pd.cut(train['FlrSF'], bins=25, labels=False)

size_df = train.groupby('size', as_index = False).agg(mean_size = ("SalePrice", "mean"))
neighbor_df = train.groupby('Neighborhood', as_index = False).agg(mean_infra = ("SalePrice", "mean"))

test = pd.read_csv("data/houseprice/test.csv")
test = test[["Id", "1stFlrSF", "2ndFlrSF", "Neighborhood"]]
test = test.assign(FlrSF = test["1stFlrSF"] + test["2ndFlrSF"])
test['size'] = pd.cut(test['FlrSF'], bins=25, labels=False)

test = pd.merge(test, size_df, how = "left", on = "size")
test = pd.merge(test, neighbor_df, how = "left", on = "Neighborhood")

test = test.assign(SalePrice = (test["mean_size"] + test["mean_infra"]) / 2)

sum(test["SalePrice"].isna())
test["SalePrice"] = test["SalePrice"].fillna(test["SalePrice"].mean())

sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df["SalePrice"] = test["SalePrice"]
sub_df.to_csv("./data/houseprice/sample_submission3.csv", index = False)

# -----------------------------------------------------------------------------


house_train= pd.read_csv("data/houseprice/train.csv")

tot = house_train.groupby('TotRmsAbvGrd') \
           .agg(counts=('LotArea', 'count'),
                mean_area=('LotArea', 'mean'),
                price = ('SalePrice', 'mean'))

# 그래프 y축 두 개로 나눠서 그리기
fig, ax1 = plt.subplots()

# 첫 번째 y축 (price)
sns.lineplot(data=tot, x='TotRmsAbvGrd', y='price', color='black', linestyle='-', ax = ax1)
ax1.set_xlabel('TotRmsAbvGrd')
ax1.set_ylabel('Price', color='black')
ax1.tick_params(axis='y', labelcolor='black')

# 두 번째 y축 (mean_area)
ax2 = ax1.twinx()  # 공유 x축을 가지는 두 번째 y축
sns.lineplot(data=tot, x='TotRmsAbvGrd', y='mean_area', color='red', linestyle='-', ax = ax2)
ax2.set_ylabel('Mean Area', color='red')
ax2.tick_params(axis='y', labelcolor='red')
fig.tight_layout()

plt.title('Price and Mean Area by Total Rooms Above Ground')
plt.show()
plt.clf()
# 방 개수당 표본 개수
house_train["TotRmsAbvGrd"].value_counts().sort_index()

#-------------------------------------------------------------------- 태현 + 가공

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

house_train = pd.read_csv("data/houseprice/train.csv")
house_train = house_train[['LotArea','TotRmsAbvGrd','SalePrice']]

house_train['TotRmsAbvGrd'].value_counts().sort_index()

# 데이터 준비
tot = house_train.groupby('TotRmsAbvGrd') \
                .agg(mean_price=('SalePrice', 'mean'),
                    mean_area=('LotArea', 'mean'))

# 그래프 그리기
fig, ax1 = plt.subplots(figsize = (3.9, 2))

# 첫 번째 y축 (mean_price)
sns.lineplot(data = tot, x='TotRmsAbvGrd', y='mean_price', color='black', linestyle='-')
ax1.set_xlabel('TotRmsAbvGrd', fontsize = 5)
ax1.set_ylabel('Mean Price', color='black', fontsize = 5)
ax1.set_xticks(np.arange(1, 15))
ax1.tick_params(axis='both', labelsize=5)

# 두 번째 y축 (mean_area)
ax2 = ax1.twinx() # 공유 x축을 가지는 두 번째 y축
sns.lineplot(data = tot, x='TotRmsAbvGrd', y='mean_area', color='red', linestyle='-')
ax2.set_ylabel('Mean Area', color='red', fontsize = 5)
ax2.tick_params(axis='y', labelcolor='red')
ax2.tick_params(axis='both', labelsize=5)

plt.title('Mean Price and Mean Area by Total Rooms Above Ground', fontsize = 5)
fig.tight_layout()

plt.show()
plt.clf()

# -------------------------------------------------------------------- 재준 현욱

house_train = pd.read_csv("data/houseprice/train.csv")
# 데이터 준비
tot = house_train.groupby('OverallQual') \
                .agg(mean_price=('SalePrice', 'mean'),
                Cond_mean=('OverallCond', 'mean'))

# 그래프 그리기
fig, ax1 = plt.subplots()

# 첫 번째 y축 (counts)
sns.lineplot(data=tot, x='OverallQual', y='mean_price', color='black', linestyle='-', ax=ax1)
ax1.set_xlabel('OverallQual')
ax1.set_ylabel('Mean_price', color='black')
ax1.tick_params(axis='y', labelcolor='black')

# 두 번째 y축 (mean_area)
ax2 = ax1.twinx() # 공유 x축을 가지는 두 번째 y축
sns.lineplot(data=tot, x='OverallQual', y='Cond_mean', color='red', linestyle='-', ax=ax2)
ax2.set_ylabel('Cond_mean', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# 그래프 제목 및 레이아웃 조정
plt.title('Mean_price and Mean Area by Total Rooms Above Ground')
fig.tight_layout()

# 그래프 표시
plt.show()
plt.clf()

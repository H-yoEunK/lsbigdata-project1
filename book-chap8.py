import pandas as pd

mpg = pd.read_csv("data/mpg.csv")
mpg.shape
mpg['drv'].unique()

import seaborn as sns
import matplotlib.pyplot as plt

plt.clf()
plt.figure(figsize = (5, 4))
sns.scatterplot(data=mpg, x = "displ", y = "hwy", hue = "drv").set(xlim = [3, 6], ylim = [10, 30])
plt.show()


df_mpg = mpg.groupby("drv", as_index = False) \
  .agg(mean_hwy=('hwy', 'mean'))
df_mpg
plt.clf()
sns.barplot(data = df_mpg.sort_values("mean_hwy"), x = "drv", y = "mean_hwy", hue = "drv")
plt.show()

df_mpg = mpg.groupby("drv", as_index = False).agg(n = ('drv' , 'count'))
sns.barplot(data = df_mpg, x = 'drv', y = 'n')
plt.show()

sns.countplot(data = mpg, x = 'drv', order = ['4', 'f', 'r'])
sns.countplot(data = mpg, x = 'drv', order = mpg['drv'].value_counts().index)
mpg


# 교재 8장, p212

import pandas as pd
import seaborn as sns

economics = pd.read_csv("./data/economics.csv")
economics.head()

economics.info()
sns.lineplot(data = economics, x = "date", y = "unemploy")
plt.show()
plt.clf()

economics["date2"] = pd.to_datetime(economics["date"])
economics.info()

economics[["date", "date2"]]
economics["date2"].dt.year
economics["date2"].dt.month
economics["date2"].dt.month_name()
economics["date2"].dt.day
economics["date2"].dt.quarter

economics["quarter"] = economics["date2"].dt.quarter
economics[["date2", "quarter"]]

# 각 날짜는 무슨 요일인가?
economics["date2"].dt.day_name()

economics["date2"] + pd.DateOffset(days=3)
import datetime
economics["date2"] + datetime.timedelta(days=3)

economics["date2"] + pd.DateOffset(months=3)
economics["date2"].dt.is_leap_year # 윤년 체크

plt.clf()
economics['year'] = economics['date2'].dt.year
sns.lineplot(data = economics, x = 'year', y = 'unemploy', errorbar = None)
# errorbar 처리한 것처럼 신뢰구간이 생기는 이유는
# datetime 같은 연도에 달이 여러 개니까 모평균을 계산해서 표시한다
# 완전 같지 않으니까 신뢰구간 표시해줌 (시계열이라 계산 방법은 조금 다르다)
plt.show()

sns.scatterplot(data = economics, x = 'year', y = 'unemploy', s = 2)

grp = economics.groupby('year', as_index = False).agg(
                mon_mean = ("unemploy", "mean"),
                mon_std = ("unemploy", "std"),
                mon_n = ("unemploy", "count")
                )

grp
grp["right_ci"] = grp['mon_mean'] + ((1.96 * grp['mon_std']) / np.sqrt(grp['mon_n']))
grp["left_ci"] = grp['mon_mean'] - ((1.96 * grp['mon_std']) / np.sqrt(grp['mon_n']))

plt.clf()
import matplotlib.pyplot as plt
x = grp["year"]
y = grp["mon_mean"]
plt.plot(x, y, color = "black")
plt.scatter(x, grp["left_ci"], color = "blue", s = 1)
plt.scatter(x, grp["right_ci"], color = "red", s = 1)
plt.show()


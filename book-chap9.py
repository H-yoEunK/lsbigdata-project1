import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# !pip install pyreadstat
raw_welfare = pd.read_spss("data/Koweps/Koweps_hpwc14_2019_beta2.sav")
welfare = raw_welfare.copy()

welfare.shape
welfare.describe()

welfare = welfare.rename(
  columns = { 'h14_g3' : 'sex',
              'h14_g4' : 'birth',
              'h14_g10' : 'marriage_type',
              'h14_g11' : 'religion',
              'p1402_8aq1' : 'income',
              'h14_eco9' : 'code_job',
              'h14_reg7' : 'code_region'})

welfare = welfare[['sex', 'birth', 'marriage_type', 'religion', 'income', 'code_job', 'code_region']]

welfare.shape

welfare["sex"].dtypes
welfare["sex"].value_counts()
welfare["sex"].isna().sum()

welfare["sex"] = np.where(welfare["sex"] == 1, 'male', 'female')

sns.countplot(data = welfare, x = "sex")
plt.show()
plt.clf()

welfare['income'].describe()
welfare['income'].isna().sum()

sex_income = welfare.dropna(subset = "income").groupby("sex", as_index = False) \ 
                    .agg(mean_income = ("income", "mean"))

sns.barplot(data = sex_income, x = "sex", y = "mean_income", hue = "sex")
plt.show()
plt.clf()

welfare['birth'].describe()
sns.histplot(data=welfare, x='birth')
plt.show()
plt.clf()

welfare["birth"].isna().sum()

welfare = welfare.assign(age = 2019 - welfare["birth"] + 1)

sns.histplot(data = welfare, x = "age")
plt.show()
plt.clf()

age_income = welfare.dropna(subset = "income").groupby("age").agg(mean_income = ("income", "mean"))
sns.lineplot(data = age_income, x = "age", y = "mean_income")
plt.show()
plt.clf()

(welfare["income"] == 0).sum()

# 나이별 income 열 na 개수 세기
my_df = welfare.assign(income_na = welfare["income"].isna()) \ 
                  .groupby("age", as_index = False) \
                  .agg(n = ("income_na", "sum")) # count로 두면 전체 사람 수가 나옴, True False 계산은 sum

sns.barplot(data = my_df, x = "age", y = "n")
plt.show()
plt.clf()

welfare = welfare.assign(ageg = np.where(welfare['age'] < 30, 'young',
                                np.where(welfare['age'] <= 59, 'middle', 'old')))

welfare['ageg'].value_counts()

sns.countplot(data = welfare, x = 'ageg')
plt.show()
plt.clf()


ageg_income = welfare.dropna(subset = 'income').groupby('ageg', as_index = False) \ 
                    .agg(mean_income = ('income', 'mean'))
                    
sns.barplot(data = ageg_income, x = 'ageg', y = 'mean_income')
sns.barplot(data = ageg_income, x = 'ageg', y = 'mean_income', order = ['young', 'middle', 'old'])

plt.show()
plt.clf()

# 나이대별 수입 분석
# cut 쓸 때 Console에 나오는 ([는 이상, 초과에 대한 구분임
# [(9, 19], (79, 89], (19, 29], ... 결과는 첫 번째 요소가 (9, 19) 범위 내에 속해 있다고 반환해주는 것

# cut
vec_x = np.random.radint(0, 100, 50)
bin_cut = np.array([0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119])
pd.cut(vec_x, bin_cut)
# age_min, age_max = (welfare['age'].min(), welfare['age'].max())
# bin_cut = [0] + [10 * i + 0 for i in np.arange(age_max //10 + 1)]

bin_cut = np.array([0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119])
welfare = welfare.assign(age_group = pd.cut(welfare["age"], 
                          bins = bin_cut, labels = np.arange(12) * 10).astype(str) + "대")
# np.version.version
# !pip install numpy --upgrade

age_income = welfare.dropna(subset = 'income').groupby('age_group', as_index = False) \ 
                    .agg(mean_income = ('income', 'mean'))

sns.barplot(data = age_income, x = "age_group", y = "mean_income")
plt.show()
plt.clf()

# pandas DataFrame을 다룰 때, 변수의 Type이 Category로 설정되어 있는 경우
# groupby + agg 가 안 먹힘
# 그래서 object Type으로 바꿔준 후 수행

welfare["age_group"] = welfare["age_group"].astype("object")
sex_age_income = welfare.dropna(subset = "income").groupby(["age_group", "sex"], as_index = False) \ 
                        .agg(mean_income = ("income", "mean"))
                        
sns.barplot(data = sex_age_income, x = 'age_group', y = 'mean_income', hue = 'sex')

# 연령대별, 성별 상위 4% 수입 찾아보기

x = np.arange(10)
np.quantile(x, q = 0.95) # 상위 5%에 해당하는 값 반환

# x에는 welfare["income"]이 들어간다
# 원래 ("income", "mean")을 하면 Python 내부에서 income 열을 추출해 평균을 구해주는데
# 우리는 정해진 mean, count, ... 가 아닌 quantile를 구하고 싶으므로 lambda를 이용해
# 사용자 정의 함수를 쉽게 넘겨줌
# def add(x, y): return x + y -> add = lambda x, y: x + y
# 무명 람다식?? 모르겠는데 이름은 뺄 수 있는 듯
# 따라서 x라는 매개변수(income)를 받아서 np.quantile(x, q = 0.96)를 실행시키는 게 다다
sex_age_income = welfare.dropna(subset = "income").groupby(["age_group", "sex"], as_index = False) \ 
                        .agg(top4per_income = ("income", lambda x: np.quantile(x, q = 0.96)))



sex_age_income = welfare.dropna(subset = "income").groupby(["age_group", "sex"], as_index = False) \ 
                        .agg(top4per_income = ("income", lambda x: np.quantile(x, q = 0.96)))

def myf(x):
  return np.quantile(x, q = 0.96)
sex_age_income = welfare.dropna(subset = "income").groupby(["age_group", "sex"], as_index = False) \ 
                        .agg(top4per_income = ("income", lambda x: myf(x)))
                      
sex_age_income

# 참고 agg에 list 넣으면 여러 개 값으로 df 받을 수 있다
welfare.dropna(subset = 'income').groupby('sex', as_index = False)['income'].agg(['mean', 'std'])

sns.barplot(data = sex_age_income, x = "age_group", y = "top4per_income", hue = "sex")
plt.show()
plt.clf()


# 9-6장

welfare["code_job"]
welfare["code_job"].value_counts()

# !pip install openpyxl
list_job = pd.read_excel("data/Koweps/Koweps_Codebook_2019.xlsx", sheet_name = "직종코드")
list_job.head()

welfare = welfare.merge(list_job, how = "left", on = "code_job")
welfare.dropna(subset = ["job", "income"])[["income", "job"]]

job_income = welfare.dropna(subset = ['job', 'income']).groupby('job', as_index = False) \ 
                    .agg(mean_income = ('income', 'mean'))

top10 = job_income.sort_values('mean_income', ascending = False).head(10)

# 대충 위에 거에 query("sex == 'female'") 붙이면 여자 거만 추려서 볼 수 있음
plt.rcParams.update({'font.family' : 'Malgun Gothic'})

sns.barplot(data= top10, y = 'job', x = 'mean_income', hue = 'job')

plt.show()
plt.clf()

bottom10 = job_income.sort_values('mean_income').head(10)

sns.barplot(data = bottom10, y = 'job', x = 'mean_income', hue = 'job').set(xlim = [0, 800])
# plt.tight_layout() 쓰면 글자가 다 보이게 그려줌 (여기서는 표가 엄청 작아진다...)
plt.show()
plt.clf()

# 9.8
welfare["marriage_type"]
df = welfare.query("marriage_type != 5").groupby("religion", as_index = False) \
            ["marriage_type"].value_counts(normalize = True) # 정규화 해줌

df.query("marriage_type == 1").assign(proportion = df["proportion"] * 100).round(1)










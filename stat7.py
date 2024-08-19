import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind

tab3 = pd.read_csv("data/tab3.csv")

tab1 = pd.DataFrame({"id": np.arange(1, 13), "score": tab3["score"]})

tab2 = tab1.assign(gender = ["female"] * 7 + ["male"] * 5)

# 1 표본 t 검정 (그룹 1개)
# 귀무가설 vs 대립가설
# H0: mu = 10 vs Ha: mu != 10
# 유의수준 5%로 설정

result = ttest_1samp(tab1['score'], popmean = 10, alternative = 'two-sided')
print(result.statistic, ", ", result.pvalue)
# t 검정통계량, 유의확률 (t-value), 여기의 p_value는 양쪽을 다 더한 값
result.df
result.confidence_interval(confidence_level = 0.95) # 95% 신뢰 구간
# 유의확률 0.0648이 유의수준 0.05보다 크므로 귀무가설을 기각하지 못한다

# 2 표본 t 검정 (그룹 2개)
# 귀무가설 vs 대립가설
# H0: mu_m = mu_f vs Ha: mu_m > mu_f
# 유의수준 1%로 설정, 두 그룹 분산 같다고 가정한다다
# 대립가설이 받아들여지지 않았을 경우에는 기존에 받아들여지던 귀무가설로 쭉 간다
# 만약 mu_m < mu_f 결과가 나와도 같다는 귀무가설 채택

f_tab2 = tab2[tab2["gender"] == "female"]
m_tab2 = tab2[tab2["gender"] == "male"]

result = ttest_ind(f_tab2["score"], m_tab2["score"], alternative = "less", equal_var = True)
# True(기본값)인 경우 동일한 모집단 분산을 가정하는 표준 독립 2 표본 테스트를 수행
# False인 경우 동일한 모집단 분산을 가정하지 않는 Welch의 t-테스트를 수행
# alternative는 대립가설 기준, "less" 의미는 첫 번째 입력 그룹의 평균이 두 번째 입력 그룹 평균보다 작다
# 분산 같은 경우: 독립 2 표본 t 검정, 분산 다른 경우: 웰치스 t 검정
result.statistic
result.pvalue
ci = result.confidence_interval(0.95)
ci[0]
ci[1]

# 대응표본 t 검정 (짝지을 수 있는 표본)
# 귀무가설 vs 대립가설
# H0: mu_before = mu_after vs Ha: mu_after > mu_before
# H0: mu_d = 0 vs Ha: mu_d > 0
# mu_d = mu_after - mu_before
# 유의수준 1%로 설정

# mu_d에 대응하는 표본으로 변환
tab3_data = tab3.pivot_table(index = 'id', columns = 'group', values = 'score').reset_index()
tab3_data['score_diff'] = tab3_data['after'] - tab3_data['before']
test3_data = tab3_data[['score_diff']]

# long_form = test3_data.melt(id_vars = 'id', value_vars = ['before', 'after'], 
#                              var_name = 'group', value_name = 'score')

# 이걸 벡터로 하는 1 표본 t test가 됨
test3_data

result = ttest_1samp(test3_data['score_diff'], popmean = 0, alternative = 'greater')
print(result.statistic, ", ", result.pvalue)
# t 검정통계량, 유의확률 (t-value), 여기의 p_value는 양쪽을 다 더한 값
result.df
result.confidence_interval(confidence_level = 0.95) # 95% 신뢰 구간

# pivot_table 실습1
df = pd.DataFrame({"id": [1, 2, 3],
                    "A": [10, 20, 30],
                    "B": [40, 50, 60]})

df_long = df.melt(id_vars = "id", value_vars = ["A", "B"], var_name = "group", value_name = "score")

df_long.pivot_table(columns = "group", values = "score")
# agg mean으로 숨겨져 있다??
# 기본에서 바꾸고 싶으면 aggfunc = 변수 사용

# pivot_table 실습2 (요일별로 펼치고 싶다)
tips = sns.load_dataset("tips")
tips_day = tips.reset_index(drop = False) \ 
                .pivot_table(index = ['index'], columns = 'day', values = 'tip').reset_index()



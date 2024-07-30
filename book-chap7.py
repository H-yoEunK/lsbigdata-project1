import pandas as pd
import numpy as np

df = pd.DataFrame({"sex": ["M", "F", np.nan, "M", "F"],
                  "score": [5, 4, 3, 4, np.nan]})

pd.isna(df) # True, False에 대한 df 반환

df.dropna(subset = "score")
df.dropna(subset = ["score", "sex"])

exam = pd.read_csv("data/exam.csv")

# 데이터 프레임 location을 사용한 인덱싱
exam.loc[[2, 7, 14], ]
exam.loc[[0], ["id", "nclass"]]
exam.iloc[0:2, 0:4]

exam.iloc[[2, 7, 4], 2] = np.nan
exam.iloc[[2, 7, 4], 2] = 3

df[df["score"] == 3.0]["score"] = 4 # 안 돼
df.loc[df["score"] == 3.0, ["score"]] = 4

# 수학 점수가 50점 이하인 학생들 점수 50점으로 상향 조정
exam.loc[exam["math"] <= 50, "math"] = 50
type(exam)

# 영어 점수 90점 이상 90점으로 하향 조정 (iloc 사용)
# iloc을 사용해서 조회하려면 무조건 숫자 벡터가 들어가야 함
exam.iloc[exam["english"] >= 90, 3] # 안 돼
exam.iloc[exam[exam["english"] >= 90].index, 3]
exam.iloc[exam[np.where(exam["english"] >= 90)[0], 3) # 튜플이라 [0] 사용해서 numpy array 꺼내오면 됨
exam.iloc[np.array(exam["english"] >= 90), 3]

exam = pd.read_csv("data/exam.csv")
# math 점수 50 이하 "-" 변경
exam.loc[exam["math"] <= 50, "math"] = "-"

# "-" 결측치를 수학 점수 평균으로 바꾸고 싶은 경우

#1
math_mean = exam.loc[(exam["math"] != "-"), "math"].mean()
exam.loc[exam['math'] == "-", 'math'] = math_mean

#2
math_mean = exam.query('math not in ["-"]')['math'].mean()
exam.loc[exam['math'] == '-', 'math'] = math_mean

#3
math_mean = exam[exam["math"] != "-"]["math"].mean()
exam.loc[exam['math'] == '-', 'math'] = math_mean

#4
exam.loc[exam['math'] == "-", ['math']] = np.nan
math_mean = exam["math"].mean()
exam.loc[pd.isna(exam['math']), ['math']] = math_mean

#5
# exam["math"] = np.where(exam["math"] == "-", (math mean 값))

vector = np.array([np.nan if x == '-' else float(x) for x in exam["math"]])
vector = np.array([float(x) if x != "-" else np.nan for x in exam["math"]])
exam["math"] = vector

math_mean = np.nanmean(np.array([np.nan if x == '-' else float(x) for x in exam["math"]]))

#6
math_mean = exam[exam["math"] != "-"]["math"].mean()
exam["math"] = exam["math"].replace("-", math_mean)
exam


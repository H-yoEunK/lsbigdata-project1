import pandas as pd
import numpy as np

# 구글 시트 불러오기기
gsheet_url = "https://docs.google.com/spreadsheets/d/1RC8K0nzfpR3anLXpgtb8VDjEXtZ922N5N0LcSY5KMx8/gviz/tq?tqx=out:csv&sheet=Sheet2"
df = pd.read_csv(gsheet_url)
df.head()

# 랜덤하게 2명을 뽑아서 보여주는 코드
np.random.seed(20240730)
np.random.choice(df["이름"], 2, replace = False)

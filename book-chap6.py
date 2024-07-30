import pandas as pd
import numpy as np

test1 = pd.DataFrame({'id': [1, 2, 3, 4, 5],
                    'midterm': [60, 80, 70, 90, 85]})
                    
test2 = pd.DataFrame({'id': [1, 2, 3, 40, 5],
                    'final': [70, 83, 65, 95, 80]})
                    
total = pd.merge(test1, test2, how = 'left', on = 'id')
total = pd.merge(test1, test2, how = 'right', on = 'id')
total = pd.merge(test1, test2, how = 'inner', on = 'id')
total = pd.merge(test1, test2, how = 'outer', on = 'id')

name = pd.DataFrame({'nclass': [1, 2, 3, 4, 5], 'teacher': ['kim', 'lee', 'park', 'choi', 'jung']})

exam = pd.read_csv("data/exam.csv")
result = pd.merge(exam, name, how = 'left', on = 'nclass')

# 세로로
score1 = pd.DataFrame({'id': [1, 2, 3, 4, 5],
                    'score': [60, 80, 70, 90, 85]})
                    
score2 = pd.DataFrame({'id': [6, 7, 8, 9, 10],
                    'score': [70, 83, 65, 95, 80]})
                    
pd.concat([score1, score2])

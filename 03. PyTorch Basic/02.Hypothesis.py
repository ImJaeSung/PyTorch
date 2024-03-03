# 어떠한 현상에 대해 이론적인 근거르 토대로 통계적 모형을 구축하며, 데이터를 수집해 해당 현상에 대한 데이터의 정확한 특성을 식별해 검증
# 귀무가설 : 처음부터 버릴 것을 예상하는 가설, 변수간의 차이나 관계가 없음
# 대립가설 : 연구가설과 동일, 연구자가 검증하려는 가설로 귀무가설을 부정하는 것을 설정하는 가설을 증명하려는 가설
# 머신러닝에서의 가설은 독립 변수와 종속변수 간의 관계를 가장 잘 근사시키기 위해 사용
# 머신러닝의 통계적 가설은 독립적인 그룹 간의 평균을 비교는 비쌍체 t-검정에 적합

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

# 성별에 따른 키 차이 검정 : 비쌍체 t-검정
man_height = stats.norm.rvs(loc = 170, scale = 10, size = 500, random_state = 1) # 특정 평균(loc)과 표준편차(scale)를 따르는 분포에서 데이터를 샘플링하는 함수
woman_height = stats.norm.rvs(loc = 150, scale = 10, size = 500, random_state = 1)

X = np.concatenate([man_height, woman_height])
Y = ["man"]*len(man_height) + ["woman"]*len(woman_height)

df = pd.DataFrame(list(zip(X, Y)), columns = ["X", "Y"])
fig = sns.displot(data = df, x = "X", hue = "Y", kind = "kde")
fig.set_axis_labels("cm", "count")
plt.show()

statistics, pvalue = stats.ttest_ind(man_height, woman_height, equal_var = True)
print("statistic : ", statistics)
print("pvalue : ", pvalue)
print("* : ", pvalue < .05)
print("** : ", pvalue < .001) # 사람의 키가 성별을 구분하는데 매우 유의미한 변수임

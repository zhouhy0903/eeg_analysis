import matplotlib.pyplot as plt
import os
import pandas as pd
from aimtrack import get_aimtrack_feature
from score import get_score

question_p=[9, 11, 13, 17, 23, 35, 36, 37]
"""
for i in range(1,60):
    try:
        first,second=get_score(i)
        aimtrack=get_aimtrack_feature(i)
        #print(len(first))
        #print(len(aimtrack))
        if len(first)!=len(aimtrack):
            question_p.append(i)
        print("  ")
    except Exception as e:
        pass
"""
df=pd.DataFrame()

for i in range(1,60):
    try:
        if i not in question_p:
            first,second=get_score(i)
            aimtrack=get_aimtrack_feature(i)
            aimtrack["score"]=first
            df=df.append(aimtrack)
    except Exception as e:
        pass

#print(question_p)
#df.to_stata("a.dta")
plt.scatter(df["xymean"],df["score"])
plt.xlabel("xymean")
plt.ylabel("score")
plt.title("score~xymean")
plt.show()
plt.scatter(df["xystd"],df["score"])
plt.show()

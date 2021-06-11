from pandas.core.frame import DataFrame
from sklearn.metrics import classification_report
import numpy as np
from ts_classifier import *
import math
import pandas as pd
from tqdm import tqdm
from state import get_state
import traceback
from eegprocess import get_raw_eeg,get_epoch_eeg
from sklearn.model_selection import train_test_split
from score import get_score
import os

def DTWDistance(s1, s2,w):
    DTW={}
    
    w = max(w, abs(len(s1)-len(s2)))
    
    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
  
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
		
    return math.sqrt(DTW[len(s1)-1, len(s2)-1])

def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):
        
        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        
        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2
    
    return math.sqrt(LB_sum)

def knn(train,test,w):
    preds=[]
    for ind,i in tqdm(enumerate(test)):
        min_dist=float('inf')
        closest_seq=[]
        #print ind
        for j in train:
            if LB_Keogh(i[:-1],j[:-1],5)<min_dist:
                dist=DTWDistance(i[:-1],j[:-1],w)
                if dist<min_dist:
                    min_dist=dist
                    closest_seq=j

        preds.append(closest_seq[-1])
    return classification_report(test[:,-1],preds)

def get_eye_feature(data_state):
    eye_move_v=[]
    for i in range(len(data_state)):
        cureye=data_state[i]
        cureye["weiyi"]=((cureye["reyex"]-cureye["reyex"].shift())**2+(cureye["reyey"]-cureye["reyey"].shift())**2+(cureye["reyez"]-cureye["reyez"].shift())**2).apply(np.sqrt)
        cureye["v"]=cureye["weiyi"]/(cureye["time"]-cureye["time"].shift())
        if len(cureye["v"])<10:
            raise Exception("not right")
        eye_move_v.append(cureye["v"].tolist()[-10:])
    return eye_move_v


def load_data():
    if os.path.exists("check.csv"):
        data=pd.read_csv("check.csv",index_col=0)
        dataxy=np.array(data)
        return dataxy

    score=[]
    eyev=[]
    highnum=0
    lownum=0
    for i in range(1,60):
        try:
            data_score=get_score(i)[0]
            data_state=get_state(i,6)
            data_eeg=get_epoch_eeg(i).drop(["condition"],axis=1)
            #data_raw_eeg=get_raw_eeg(i)
            print(len(data_state))
            print(len(data_score))
            print(len(data_eeg["epoch"].value_counts()))
            if not (len(data_score)==len(data_eeg["epoch"].value_counts()) and 
            len(data_score)==len(data_state)): continue
            eyemovev=get_eye_feature(data_state)
            for j in range(len(data_score)):
                if data_score[j]>10:
                    score.append(1)
                    highnum+=1
                    eyev.append(eyemovev[j])
                if data_score[j]<6.5:
                    score.append(0)
                    lownum+=1
                    eyev.append(eyemovev[j])
            
        except Exception as e:
            traceback.print_exc()
            pass
    
    eye_regress=pd.DataFrame(eyev)
    eye_regress["score"]=score
    eye_regress.fillna(0,inplace=True)
    dataxy=np.array(eye_regress)
    eye_regress.to_csv("check.csv")
    print(highnum,lownum)
    return dataxy

data=load_data()
print(data[:,-1].shape)
print(data[:,-1])

from sklearn.model_selection import KFold
 
kf = KFold(n_splits=10)
#2折交叉验证，将数据分为两份即前后对半分，每次取一份作为test集
for train_index, test_index in kf.split(data):
    #train_index与test_index为下标
    train_X = data[train_index]
    test_X= data[test_index]
    print(knn(train_X,test_X,4))

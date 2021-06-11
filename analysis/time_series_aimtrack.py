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
from aimtrack import get_aimtrack
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

def get_aimtrack_feature(aimtrack):
    feature=[]
    print(aimtrack)
    aimtrack_num=len(aimtrack[aimtrack["exp_num"]==1]["shoot_num"].value_counts())
    print(aimtrack_num)
    for i in range(aimtrack_num):
        curaimtrack=aimtrack[aimtrack["exp_num"]==1][aimtrack["shoot_num"]==i+1]
        curaimtrack["v"]=((curaimtrack["x"]-curaimtrack["x"].shift())**2+(curaimtrack["y"]-curaimtrack["y"].shift())**2).apply(np.sqrt)
        curaimtrack["x_y"]=curaimtrack["x"]+curaimtrack["y"]
        curaimtrack_pad=[]
        if len(curaimtrack)<20:
            curaimtrack_pad=[0]*(20-len(curaimtrack))+curaimtrack["x_y"].tolist()
        else:
            curaimtrack_pad=curaimtrack["x_y"].tolist()[-20:]
        print(len(curaimtrack_pad))
        feature.append(curaimtrack_pad)
    #return eye_move_v
    return feature

def load_data():
    if os.path.exists("check_aimtrack.csv"):
        data=pd.read_csv("check_aimtrack.csv",index_col=0)
        dataxy=np.array(data)
        return dataxy

    score=[]
    aimtrack_all=[]
    highnum=0
    lownum=0
    for i in range(1,60):
        try:
            data_score=get_score(i)[0]
            data_state=get_state(i,6)
            data_eeg=get_epoch_eeg(i).drop(["condition"],axis=1)
            data_aimtrack=get_aimtrack(i)
            aimtrack_feature=get_aimtrack_feature(data_aimtrack)
            
            if not (len(data_score)==len(data_eeg["epoch"].value_counts()) and 
            len(data_score)==len(data_state) and len(aimtrack_feature)==len(data_score)): continue
            for j in range(len(data_score)):
                if data_score[j]>10:
                    score.append(1)
                    highnum+=1
                    aimtrack_all.append(aimtrack_feature[j])
                if data_score[j]<6.5:
                    score.append(0)
                    lownum+=1
                    aimtrack_all.append(aimtrack_feature[j])
            
        except Exception as e:
            traceback.print_exc()
            pass
    aim=pd.DataFrame(aimtrack_all)
    aim["score"]=score
    aim.fillna(0,inplace=True)
    dataxy=np.array(aim)
    aim.to_csv("check_aimtrack.csv")
    print(highnum,lownum)
    return dataxy

data=load_data()
print(data[:,-1].shape)
print(data[:,-1])


#x_train, x_test, y_train, y_test = train_test_split(data[:,:-1],data[:,-1],test_size = 0.1)



from sklearn.model_selection import KFold
 
kf = KFold(n_splits=10)
#2折交叉验证，将数据分为两份即前后对半分，每次取一份作为test集
for train_index, test_index in kf.split(data):
    #train_index与test_index为下标
    train_X = data[train_index]
    test_X= data[test_index]
    knn_model=knn(train_X,test_X,4)
    print(knn_model)
#print("train_X",train_X)
#print("test_X",test_X)

"""
print(x_train.shape,x_test.shape)
train=np.column_stack((x_train,y_train))
test=np.column_stack((x_test,y_test))
print(train.shape,test.shape)
#print(train.shape)
#print(test.shape)
print(knn(train,test,4))
"""

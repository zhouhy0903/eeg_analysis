from glob import glob
import os
from score import get_score
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from state import get_state
import traceback
from eegprocess import get_raw_eeg,get_epoch_eeg

def get_eye_feature(data_state):
    eye_move_v_list=[]
    for i in range(len(data_state)):
        cureye=data_state[i]
        cureye["weiyi"]=((cureye["reyex"]-cureye["reyex"].shift())**2+(cureye["reyey"]-cureye["reyey"].shift())**2+(cureye["reyez"]-cureye["reyez"].shift())**2).apply(np.sqrt)
        cureye["v"]=cureye["weiyi"]/(cureye["time"]-cureye["time"].shift())
        #print(cureye)
        threshold=cureye["v"].mean()
        """
        plt.plot(cureye["v"].tolist()) 
        plt.show()
        """
        cureye_eye_move=cureye[cureye["v"]>threshold]
        #print(cureye_eye_move)
        #eye_move_v=cureye_eye_move["v"].mean()
        eye_move_v=[]
        start=[j*100 for j in range(10)]
        end=[(j+1)*100 for j in range(10)]
        for j in range(len(start)):
            temp=len(cureye_eye_move[cureye_eye_move["v"]>start[j]][cureye_eye_move["v"]<end[j]])
            if np.isnan(temp):
                raise Exception("found none")
            eye_move_v.append(temp)

        eye_move_v_list.append(eye_move_v)
        

    #print(eye_move_v_list)
    return eye_move_v_list



# get_eyedata
def get_eyedata():
    high_eye_feature=[]
    low_eye_feature=[]
    all_eye_feature=[]
    all_score=[]
    for i in range(1,60):
        try:
            data_score=get_score(i)[0]
            data_state=get_state(i,3)
            data_eeg=get_epoch_eeg(i).drop(["condition"],axis=1)
            #data_raw_eeg=get_raw_eeg(i)
            print(len(data_state))
            print(len(data_score))
            print(len(data_eeg["epoch"].value_counts()))
            if not (len(data_score)==len(data_eeg["epoch"].value_counts()) and 
            len(data_score)==len(data_state)): continue
            eye_move=get_eye_feature(data_state)
            all_eye_feature.extend(eye_move)
            all_score.extend(data_score)
            for j in range(len(data_score)):
                #if np.isnan(eye_move[j]): continue
                print(eye_move[j])
                if eye_move[j]==None: 
                    print("yes"*10)
                    continue
                if data_score[j]>9:
                    high_eye_feature.append(eye_move[j])
                if data_score[j]<8:
                    low_eye_feature.append(eye_move[j])
                
        except Exception as e:
            traceback.print_exc()
            pass
    all_eye_feature_np=np.array(all_eye_feature)
    print(all_eye_feature_np.shape)
    print(np.array(all_score).shape)
    eye_regress=pd.DataFrame(all_eye_feature,columns=[str(i) for i in range(10)])
    eye_regress["score"]=all_score
    eye_regress.to_csv("eye_feature_regress.csv")
    print(eye_regress)
    return high_eye_feature,low_eye_feature


def ttest_eye(high,low):
    feature_num=len(high[0])
    print("Feature num is {}".format(feature_num))
    pvalue=[]
    for feature_idx in range(feature_num):
        fhigh=[]
        flow=[]
        for i in range(len(high)):
            fhigh.append(high[i][feature_idx]) 
        for i in range(len(low)):
            flow.append(low[i][feature_idx])
        print(len(fhigh),len(flow))
        result=stats.ttest_ind(a=fhigh,b=flow)
        print(result)
        pvalue.append(result[1])

    print(pvalue)
    pvalue_df=pd.DataFrame(pvalue)

    pvalue_df.to_csv("ttest_eye.csv")

def csv_to_dta(path="eye_feature_regress.csv"):
    data=pd.read_csv(path,index_col=0)
    data.to_stata("a.dta")


if __name__=="__main__":

    #high,low=get_eyedata()
    csv_to_dta()
    """
    high_df=pd.DataFrame(high)
    high_df.to_csv("high.csv")
    low_df=pd.DataFrame(low)
    low_df.to_csv("low.csv")

    high=pd.read_csv("high.csv",index_col=0).values.tolist()
    low=pd.read_csv("low.csv",index_col=0).values.tolist()



    ttest_eye(high,low)
    """
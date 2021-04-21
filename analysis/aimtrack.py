from glob import glob
import os
import pandas as pd
import configparser
import warnings
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

files=[]
save_droprepeat_path=""
save_pic_path=""
save_feature_path=""
ifdroprepeat=0

def init():
    global files,save_droprepeat_path,save_pic_path,save_feature_path,ifdroprepeat
    cf=configparser.ConfigParser()
    cf.read("aimtrack_path.conf")
    path_names=cf.options("path")
    aimtrack_path_split=cf.get("path","aimtrack_path_split")    
    save_droprepeat_path=cf.get("path","outcome_path_droprepeat")
    save_pic_path=cf.get("path","outcome_path_aimtrack")
    save_feature_path=cf.get("path","outcome_path_features")

    ifdroprepeat=cf.get("path","droprepeat")

    files=glob(os.path.join(aimtrack_path_split,"*.csv"))


def drop_repeat():
    global files,save_droprepeat_path
    print(save_droprepeat_path)
    if not os.path.exists(save_droprepeat_path):
        os.makedirs(save_droprepeat_path)
    for f in tqdm(files):
        data=pd.read_csv(f)
        data.columns=["exp_num","shoot_num","x","y"]
        newdata=pd.DataFrame()
        for i in range(2):
            for j in range(30):
                cur_data=data[data["exp_num"]==i+1][data["shoot_num"]==j+1]
                if len(cur_data)==0: continue
                cur_data["x_last"]=cur_data["x"].shift()
                cur_data["y_last"]=cur_data["y"].shift()
                cur_data.drop(cur_data[cur_data["x"]==cur_data["x_last"]][cur_data["y"]==cur_data["y_last"]].index,inplace=True)
                newdata=newdata.append(cur_data)
        try:
            newdata[["exp_num","shoot_num","x","y"]].to_csv(os.path.join(save_droprepeat_path,re.findall(f".*\\\\(.*)",f)[0]),index=False)
        except Exception as e:
            print("not find file name!")        

def show_aimtrack():
    global files,save_pic_path,save_droprepeat_path
    files=glob(os.path.join(save_droprepeat_path,"*.csv"))

    if not os.path.exists(save_pic_path):
        os.makedirs(save_pic_path)

    for f in tqdm(files):
        data=pd.read_csv(f)
        for i in range(1):
            for j in range(30):
                cur_data=data[data["exp_num"]==i+1][data["shoot_num"]==j+1]
                if len(cur_data)==0: continue
                plt.plot(cur_data["x"],cur_data["y"])
                plt.scatter(cur_data["x"].iloc[0],cur_data["y"].iloc[0],c="lightskyblue",label="start")
                plt.scatter(cur_data["x"].mean(),cur_data["y"].mean(),c="deepskyblue",label="mean")
                plt.scatter(cur_data["x"].iloc[-1],cur_data["y"].iloc[-1],c="blue",label="end")
                plt.scatter(0,0,c="red",label="center")
                plt.title("{}-{}".format(i+1,j+1))
                plt.legend()
                plt.show()
        break
    
def save_aimtrack_feature():
    global files,save_pic_path,save_droprepeat_path,save_feature_path
    if not os.path.exists(save_feature_path):
        os.makedirs(save_feature_path)
    files=glob(os.path.join(save_droprepeat_path,"*.csv"))
    columns=["xmean","ymean","xymean","xstd","ystd","xystd"]
    for f in tqdm(files):
        data=pd.read_csv(f)
        feature=[]
        for i in range(1):
            for j in range(30):
                cur_data=data[data["exp_num"]==i+1][data["shoot_num"]==j+1]
                if len(cur_data)==0: continue
                feature.append([cur_data["x"].mean(),cur_data["y"].mean(),(cur_data["x"]**2+cur_data["y"]**2).apply(np.sqrt).mean(),
                               cur_data["x"].std(),cur_data["y"].std(),(cur_data["x"]**2+cur_data["y"]**2).apply(np.sqrt).std()])
        feature_df=pd.DataFrame(feature,columns=columns)
        try:
            feature_df.to_csv(os.path.join(save_feature_path,re.findall(f".*\\\\(.*)",f)[0]),index=False)
        except Exception as e:
            print("save aimtrack feature error!")

def get_aimtrack_feature(num):
    global save_feature_path
    num-=1
    feature_files=glob(os.path.join(save_feature_path,"*.csv"))
    print("Reading aimtrack feature from {}".format(feature_files[num]))
    data=pd.read_csv(feature_files[num])
    return data

    
init()
if __name__=="__main__":
    if int(ifdroprepeat)==0:
        print("droprepeat")
        drop_repeat()
    print(save_feature_path)
    # show_aimtrack()
    # save_aimtrack_feature()
    print(get_aimtrack_feature(1))
    


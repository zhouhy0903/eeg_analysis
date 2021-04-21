from glob import glob
import os
import pandas as pd
import configparser
import warnings
import re
from tqdm import tqdm

warnings.filterwarnings("ignore")

files=[]
save_path=[]

def init():
    global files,save_path
    cf=configparser.ConfigParser()
    cf.read("path.conf")
    path_names=cf.options("path")
    aimtrack_path_split=cf.get("path","aimtrack_path_split")
    
    save_path=cf.get("path","outcome_path")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    files=glob(os.path.join(aimtrack_path_split,"*.csv"))


def drop_repeat():
    global files,save_path
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
            newdata[["exp_num","shoot_num","x","y"]].to_csv(os.path.join(save_path,re.findall(f".*\\\\(.*)",f)[0]),index=False)
        except Exception as e:
            print("not find file name!")        

if __name__=="__main__":
    init()
    drop_repeat()
import os
import configparser
from glob import glob
import pandas as pd
import warnings
import numpy as np
import re

warnings.filterwarnings("ignore")
files=[]
save_path=""
save_path_state=""
save_path_marker=""
def init():
    global files,save_path,save_path_state,save_path_marker
    cf=configparser.ConfigParser()
    cf.read("state_path.conf")
    path_state=cf.get("path","path_state")
    save_path=cf.get("path","path_save")
    save_path_marker=os.path.join(save_path,"marker")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path_state=os.path.join(save_path,"state")
    #path_state_split=cf.get("path","p")
    files=glob(os.path.join(path_state,"*.csv"))


def save_state():
    global save_path_state
    if not os.path.exists(save_path_state):
        os.makedirs(save_path_state)
    
    for f in files[1:]:
        print(f)
        with open(f,"rb") as fi:
            lines=fi.readlines()
            flag=0
            for i,line in enumerate(lines):
                if line.decode("utf-8").startswith("liangba"): 
                    flag=i
                    break
            lines=lines[:flag]
            print(f)
            with open(os.path.join(save_path_state,re.findall(r".*\\(.*)",f)[0]),"wb") as fo:
                for l in lines[2:]:
                    fo.write(l)

def get_state():
    global save_path_state,save_path_marker
    if not os.path.exists(save_path_marker):
        os.makedirs(save_path_marker)
    files_presave=glob(os.path.join(save_path_state,"*.csv"))

    for f in files_presave:
        data=pd.read_csv(f)[["time","markerText"]]
        data=data[~pd.isnull(data["markerText"])]
        data=data.reset_index(drop=True)
        start_first_shoot=data[data["markerText"]=="StartNoEmotionalShot"].index[0]
        stop_first_shoot=data[data["markerText"]=="StopNoEmotionalShot"].index[0]
        start_second_shoot=data[data["markerText"]=="StartVRVideoShot1"].index[0]
        stop_second_shoot=data[data["markerText"]=="StopVRVideoShot1"].index[0]

        new=pd.DataFrame()
        new=new.append(data[start_first_shoot:stop_first_shoot]).append(data[start_second_shoot:stop_second_shoot])
        #print(data[data["time"]>startshoot])

        #for i in range(len(data)):
        #    print(data.iloc[i]["markerText"])
        #print(re.findall(r".*\\(.*)",f))
        new.to_csv(os.path.join(save_path_marker,re.findall(r".*\\(.*)",f)[0]),index=False)
init()

if __name__=="__main__":
    #save_state()
    get_state()
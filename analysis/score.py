import pandas as pd
import os
from glob import glob
import configparser
import warnings
import re
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

files=[]

def init():
    global files
    cf=configparser.ConfigParser()
    cf.read("score_path.conf")
    path_score=cf.get("path","path_score")
    files=glob(os.path.join(path_score,"*.csv"))

def get_score(num):
    global files
    num-=1
    print("Reading scores from {}".format(files[num]))
    data=pd.read_csv(files[num],index_col=0)
    data.columns=["first","second","nothing"]
    data.drop(["nothing"],axis=1,inplace=True)
    data.index.name="num"
    first=data[data["first"]!=-1]["first"].tolist()
    second=data[data["second"]!=-1]["second"].tolist()
    return first,second

init()
if __name__=="__main__":
    pass
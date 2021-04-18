import mne
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import pandas as pd
from glob import glob
import re
import warnings
from mne.preprocessing import ICA
from mne.preprocessing import corrmap

warnings.filterwarnings("ignore")

data_path=r'D:\code\code\eegemotion\test'
save_path=r'D:\code\code\eegemotion\test\outcome'

low_pass=0.5
high_pass=45
componum=31 # ica component
ica_threshold=0.7 # 0~1 

filename=[]
mneraw=[]
icas=[]
ica_pic_path=""
eeg_afterica_path=""
def init():
    global save_path,ica_pic_path,eeg_afterica_path

    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ica_pic_path=os.path.join(save_path,"pic")
    eeg_afterica_path=os.path.join(save_path,"eeg_removeeye")
    if not os.path.exists(ica_pic_path):
        os.mkdir(ica_pic_path)
    if not os.path.exists(eeg_afterica_path):
        os.mkdir(eeg_afterica_path)
    

def read_raw_file(path):
    global filename
    global mneraw
    if not path:
        raise Exception("Path None")
    path=str(Path(path) / "*.vhdr")
    files=glob(path)
    if len(files)==0:
        raise Exception("not have vhdr file")
    for f in files:
        filename.append(f[len(data_path)+1:-5])

    
    for f in files:
        raw=mne.io.read_raw(f,preload=True)
        raw=raw.set_montage("standard_1020")
        mneraw.append(raw)
        print(raw.info)

def filter(low_pass,high_pass):
    global mneraw
    for raw in mneraw:
        raw=raw.filter(low_pass,high_pass)

def get_ica():
    global mneraw
    global componum
    global ica_pic_path
    global filename
    global icas
    print(ica_pic_path)


    for i,raw in enumerate(mneraw):
        ica=ICA(n_components=componum,random_state=90)
        ica.fit(raw)
        icas.append(ica)
        fig=ica.plot_components(show=False)

        cur_ica_path=os.path.join(ica_pic_path,filename[i])
        print(cur_ica_path)
        if not os.path.exists(cur_ica_path):
            os.mkdir(cur_ica_path)
        for j in range(len(fig)):
            fig[j].savefig(os.path.join(cur_ica_path,"ica{}.jpg".format(j+1)))
        

def find_pattern():
    global mneraw,icas,ica_pic_path

    raw=mneraw[0]
    ica=icas[0]

    fig1,fig2=corrmap(icas,template=(0,0),threshold=ica_threshold,show=False,label="blink")
    
    fig1.savefig(os.path.join(ica_pic_path,"blink_pattern.jpg"))
    print(type(fig2))
    if type(fig2)==list:
        for j in range(len(fig2)):
            fig2[j].savefig(os.path.join(ica_pic_path,"blink_found_{}.jpg".format(j+1)))
    else:
        fig2.savefig(os.path.join(ica_pic_path,"blink_found.jpg"))


def remove_ica():
    global mneraw,icas,eeg_afterica_path,filename

    for i in range(len(icas)):
        icas[i].exclude=icas[i].labels_["blink"]
        icas[i].apply(mneraw[i])
        mneraw[i].save(os.path.join(eeg_afterica_path,filename[i]+".fif"),overwrite=True)

def check():
    global mneraw
    for raw in mneraw:
        print(raw.info)
        break


init()
print(ica_pic_path)
read_raw_file(data_path)
filter(low_pass,high_pass)
get_ica()
find_pattern()
remove_ica()
#check()
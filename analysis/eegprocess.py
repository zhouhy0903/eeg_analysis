import mne
from glob import glob
import os
import configparser
import mne
import matplotlib.pyplot as plt
from score import get_score
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
import numpy as np
import traceback
import pandas as pd
import re

files=[]
path_eeg=""
path_eeg_save_h5=""

def init():
    global files,path_eeg,path_eeg_save_h5
    cf=configparser.ConfigParser()
    cf.read("eeg_path.conf")
    path_names=cf.options("path")
    path_eeg=cf.get("path","path_eeg")
    path_eeg_save_h5=os.path.join(path_eeg,"eeg2h5")
    files=glob(os.path.join(path_eeg,"*.fif"))
    

def preprocess():
    global files,path_eeg_save_h5
    if not os.path.exists(path_eeg_save_h5):
        os.makedirs(path_eeg_save_h5)

    
    count=0
    for i,f in enumerate(files):
        try:
            print(i)
            print("Reading eeg from {}".format(f))
            data=mne.io.read_raw_fif(f)
            data=data.set_montage("standard_1020")
            events,event_id=mne.events_from_annotations(data)
            epochs=mne.Epochs(data,events,event_id,tmin=-10,tmax=0,event_repeated='drop',preload=True)
            first,second=get_score(i+1)
            rest_eeg=epochs["s1001"][0]
            total_shot_num=len(first)+len(second)
            first_shoot_eeg=epochs["s1002"][-total_shot_num:-len(second)]
            second_shoot_eeg=epochs["s1002"][-len(second):]
            fc=0
            sc=0

            for j in range(len(epochs)-1,1,-1):
                if epochs.events[j][2]==4:
                    sc+=1
                else:
                    if sc>0: break
            for k in range(j,1,-1):
                if epochs.events[k][2]==6: continue 
                else: break
            for j in range(k,1,-1):
                if epochs.events[j][2]==4:
                    fc+=1
                else:
                    if fc>0: break
            print(len(first_shoot_eeg))
            print(fc)
            first_pd=first_shoot_eeg.to_data_frame()
            second_pd=second_shoot_eeg.to_data_frame()
            cur_path=os.path.join(path_eeg_save_h5,re.findall(r".*\\(.*?)\.[a-z]",f)[0])
            epoch_min=min(first_pd["epoch"])
            first_pd["epoch"]=first_pd["epoch"].apply(lambda x:x-epoch_min)

            first_pd.to_hdf(cur_path+".h5",key="shoot_eeg")
        except Exception as e:
            traceback.print_exc()
            pass

def get_raw_eeg(num):
    global files
    data=mne.io.read_raw_fif(files[num-1])
    data=data.set_montage("standard_1020")
    return data    


def get_epoch_eeg(num):
    global path_eeg_save_h5 
    if len(path_eeg_save_h5)=="":
        preprocess()
    epoch_files=glob(os.path.join(path_eeg_save_h5,"*.h5"))
    print("Reading from {}".format(epoch_files[num-1]))
    data=pd.read_hdf(epoch_files[num-1],key="shoot_eeg")
    return data

init()
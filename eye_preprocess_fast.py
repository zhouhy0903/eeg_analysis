import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

path="D:\data\eegemotion\preprocess\state-P\state"
files_name=os.listdir(path)
#print(files_name)

path_shoot_time="D:\data\eegemotion\preprocess\eeg-N\shoottime"
path_eyemove_save="D:\data\eegemotion\preprocess\eye-P-first-fast"
shoot_time_name=os.listdir(path_shoot_time)
current=0

for file_name in files_name[current:]:
    print(file_name)
    eyedata=pd.read_csv(os.path.join(path,file_name))
    #0 time;1 nothing;2 event_id
    #event_id:10003:s1002 shoot marker
    eyedata_epochs=pd.DataFrame(columns=["t","elx","ely","elz","erx","ery","erz","elt","ert","markerText"])
    time_begin=eyedata.iloc[0]["time"]
    for j in tqdm(range(len(eyedata))):
        time_rel=eyedata.iloc[j]["time"]-time_begin
        elx,ely,elz,erx,ery,erz,gpx,gpy,gpz,grx,gry,grz,elt,ert,mrktxt=[np.nan for i in range(15)]
        if not pd.isnull(eyedata["leftEyeTargetPosition"].iloc[j]):
            elx,ely,elz=tuple(eval(eyedata["leftEyeTargetPosition"].iloc[j]))
        if not pd.isnull(eyedata["rightEyeTargetPosition"].iloc[j]):
            erx,ery,erz=tuple(eval(eyedata["rightEyeTargetPosition"].iloc[j]))
        if not pd.isnull(eyedata["leftEyeTarget"].iloc[j]):
            elt=eyedata["leftEyeTarget"].iloc[j]
        if not pd.isnull(eyedata["rightEyeTarget"].iloc[j]):
            ert=eyedata["rightEyeTarget"].iloc[j]
        if not pd.isnull(eyedata["markerText"].iloc[j]):
            mrktxt=eyedata["markerText"].iloc[j]
        s = pd.Series({"t":time_rel,"elx":elx,"ely":ely,"elz":elz,"erx":erx,"ery":ery,"erz":erz,"elt":elt,"ert":ert,"markerText":mrktxt})
        eyedata_epochs=eyedata_epochs.append(s,ignore_index=True)

    eyedata_epochs.to_csv(os.path.join(path_eyemove_save,file_name))


import matplotlib.pyplot as plt
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D

path="D:\data\eegemotion\preprocess\state-N\state"
files_name=os.listdir(path)
#path_eeg_time="D:\data\eegemotion\preprocess\eeg-N\eventtime"
#path_shoot_time="D:\data\eegemotion\preprocess\eeg-N\shoottime"
path_eeg_time="D:\data\eegemotion\preprocess\eeg-P\eventtime"
path_shoot_time="D:\data\eegemotion\preprocess\eeg-P\shoottime"
eeg_time_name=os.listdir(path_eeg_time)
#question=["N009_hancheng_2020-12-05_08-45-30.csv","N037_houzhenqiang_2020-12-16_11-29-27.csv"]
question=[]
for i,file_name in enumerate(files_name):
    if eeg_time_name[i] in question:
        continue

    print(file_name,eeg_time_name[i])
    
    eventtime=pd.read_csv(os.path.join(path_eeg_time,eeg_time_name[i]),index_col=0)
    print(eventtime[eventtime["2"]==10003])

    eyedata=pd.read_csv(os.path.join(path,file_name))
    eyex,eyey,eyez,eyet=[],[],[],[]
    #0 time;1 nothing;2 event_id
    #event_id:10003:s1002 shoot marker
    shoot_time_first,shoot_time_second=[],[]
    count=0

    j=0
    while j<len(eventtime):
        if eventtime["2"].iloc[j]==10003:
            while j+1<len(eventtime) and (eventtime["2"].iloc[j]==10003 or eventtime["2"].iloc[j]==10002):
                if eventtime["2"].iloc[j]==10003:
                    if count==1:
                        shoot_time_first.append(eventtime["0"].iloc[j])
                    if count==2:
                        shoot_time_second.append(eventtime["0"].iloc[j])
                j+=1
            count+=1
        j+=1

    shoot_time_first.append(-1)
    print(shoot_time_first)
    shoot_time=shoot_time_first+shoot_time_second
    print(shoot_time)
    if len(shoot_time)<30:
        raise("not enough shoot")

    shoot_time=pd.DataFrame(shoot_time)
    shoot_time.to_csv(os.path.join(path_shoot_time,eeg_time_name[i]))

    #time_begin=eyedata.iloc[i]["time"]
    #for j in range(len(eyedata)):
    #    data=eyedata["rightEyeTargetPosition"].iloc[j]
    #    if not pd.isnull(data):
    #        time_rel=eyedata.iloc[j]["time"]-time_begin
    #        x,y,z=tuple(eval(data))
    #        eyet.append(time_rel)
    #        eyex.append(x)
    #        eyey.append(y)
    #        eyez.append(z)
    

    #print(len(eyedata),len(eyex))
    #fig=plt.figure()
    #eyet=20
    #ax2 = Axes3D(fig)
    #ax2.plot(eyex[:eyet],eyey[:eyet],eyez[:eyet])
    #plt.show()


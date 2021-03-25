import matplotlib.pyplot as plt
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

path="D:\data\eegemotion\preprocess\state-N\state"
files_name=os.listdir(path)
path_shoot_time="D:\data\eegemotion\preprocess\eeg-N\shoottime"
path_eyemove_save="D:\data\eegemotion\preprocess\eye-N-first"
shoot_time_name=os.listdir(path_shoot_time)

for i,file_name in enumerate(files_name):
    print(file_name,shoot_time_name[i])
    eventtime=pd.read_csv(os.path.join(path_shoot_time,shoot_time_name[i]),index_col=0)
    print(eventtime.head())
    eyedata=pd.read_csv(os.path.join(path,file_name))
    eyex,eyey,eyez,eyet=[],[],[],[]
    eyex_data,eyey_data,eyez_data,eyet_data=[],[],[],[]
    #0 time;1 nothing;2 event_id
    #event_id:10003:s1002 shoot marker
    eyedata_epochs=pd.DataFrame(columns=["shoot","x","y","z","t"])


    time_begin=eyedata.iloc[i]["time"]
    count=0
    exp_num=1
    shoot_num=1
    flag=0
    for j in tqdm(range(len(eyedata))):
        data=eyedata["rightEyeTargetPosition"].iloc[j]
        if not pd.isnull(data):
            time_rel=eyedata.iloc[j]["time"]-time_begin
            if (time_rel+5)*1000>eventtime["0"].iloc[count]:
                x,y,z=tuple(eval(data))
                shootdata=str(exp_num)+"-"+str(shoot_num)
                s = pd.Series({'shoot':shootdata, 'x':x,'y':y,'z':z,'t':time_rel})
                eyedata_epochs=eyedata_epochs.append(s,ignore_index=True)
                #eyet_data.append(time_rel)
                #eyex_data.append(x)
                #eyey_data.append(y)
                #eyez_data.append(z)
                flag=1
            
            if flag==1 and time_rel*1000>eventtime["0"].iloc[count]: 
                #eyet.append(eyet_data)
                #eyex.append(eyex_data)
                #eyey.append(eyey_data)
                #eyez.append(eyez_data)
                #eyex_data,eyey_data,eyez_data,eyet_data=[],[],[],[]
                
                count+=1
                if count>=len(eventtime):
                    break
                shoot_num+=1
                if eventtime["0"].iloc[count]==-1: 
                    count+=1
                    exp_num=2
                    shoot_num=1
                flag=0


    eyedata_epochs.to_csv(os.path.join(path_eyemove_save,file_name))
    break
    #fig=plt.figure()
    #eyet=20
    #ax2 = Axes3D(fig)
    #ax2.plot(eyex[:eyet],eyey[:eyet],eyez[:eyet])
    #plt.show()


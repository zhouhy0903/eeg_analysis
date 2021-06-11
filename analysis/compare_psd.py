import pandas as pd
import os
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import re
from tqdm import tqdm

def get_value(column_name,lists):
    answer=[]
    for l in lists:
        if l[0] in column_name:
            answer.append(l[1])
    return answer


def compare_3s_before_shot():
    highpath=r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\single_csv\high*.csv"
    lowpath=r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\single_csv\low*.csv"
    athletepath=r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\single_csv\athlete*.csv"
    save_path=r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\single_csv"
    highfiles=glob(highpath)
    lowfiles=glob(lowpath)
    athletefiles=glob(athletepath)
    column_names=[['F3', 'Fz', 'F4'], 
                ['F3', 'Fz', 'F4'], 
                ['F3', 'Fz', 'F4','P7', 'P3', 'P4', 'P8','O1','O2','M1'], 
                ['F3', 'Fz', 'F4','P7', 'P3', 'P4', 'P8','O1','O2','M1'], 
                ['F3', 'Fz', 'F4','P7', 'P3', 'P4', 'P8','O1','O2','M1']]
    bands = [(0, 4, '1Delta (0-4 Hz)'), (4, 8, '2Theta (4-8 Hz)'),
            (8, 12, '3Alpha (8-12 Hz)'), (12, 30, '4Beta (12-30 Hz)'),
            (30, 45, '5Gamma (30-45 Hz)')]


    for i in range(5):
        high=pd.read_csv(highfiles[i])
        low=pd.read_csv(lowfiles[i])
        athlete=pd.read_csv(athletefiles[i])
        
        x=np.arange(len(column_names[i]))
        highpsd=get_value(column_names[i],high[['names','psd']].values.tolist())
        lowpsd=get_value(column_names[i],low[['names','psd']].values.tolist())
        athletepsd=get_value(column_names[i],athlete[['names','psd']].values.tolist())
        wid=0.2
        plt.bar(x-wid,highpsd,width=wid/3*2,align="center",label="high") 
        plt.bar(x,lowpsd,width=wid/3*2,align="center",label="low") 
        plt.bar(x+wid,athletepsd,width=wid/3*2,align="center",label="athlete") 
        plt.xticks(x, labels=column_names[i])
        plt.legend()
        plt.title(bands[i][2]+" psd")
        plt.savefig(os.path.join(save_path,bands[i][2]+"_psd.jpg"))
        plt.close()
    print(highfiles,lowfiles,athletefiles)

def compare_3s_12epoch_psd():
    highpath=r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\csv_3"
    lowpath=r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\csv_3"
    athletepath=r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\athlete\csv_3"
    save_path=r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\outcome\psd_change"
    highfiles=glob(highpath+"\*")
    lowfiles=glob(lowpath+"\*")
    athletefiles=glob(athletepath+"\*")
    column_names=[['F3', 'Fz', 'F4'], 
                ['F3', 'Fz', 'F4'], 
                ['F3', 'Fz', 'F4','P7', 'P3', 'P4', 'P8','O1','O2','M1'], 
                ['F3', 'Fz', 'F4','P7', 'P3', 'P4', 'P8','O1','O2','M1'], 
                ['F3', 'Fz', 'F4','P7', 'P3', 'P4', 'P8','O1','O2','M1']]
    bands = [(0, 4, 'Delta (0-4 Hz)'), (4, 8, 'Theta (4-8 Hz)'),
            (8, 12, 'Alpha (8-12 Hz)'), (12, 30, 'Beta (12-30 Hz)'),
            (30, 45, 'Gamma (30-45 Hz)')]
    

    highpsd_all,lowpsd_all,athletepsd_all=[[] for i in range(5)],[[] for i in range(5)],[[] for i in range(5)]
    for hf in reversed(highfiles):
        start,end=[float(x) for x in re.findall(r".*\\(.*)_(.*)",hf)[0]]
        print("Reading data from {} to {} befor shot".format(start,end))
        curfile=re.findall(r".*\\(.*)",hf)[0]

        curhighpath=glob(os.path.join(highpath,curfile,"high*.csv"))
        curlowpath=glob(os.path.join(lowpath,curfile,"low*.csv"))
        curathletepath=glob(os.path.join(athletepath,curfile,"athlete*.csv"))
        
        for i in tqdm(range(len(column_names))):
                
            highdata=pd.read_csv(curhighpath[i])
            lowdata=pd.read_csv(curlowpath[i])
            athletedata=pd.read_csv(curathletepath[i])

            highpsd=get_value(column_names[i],highdata[['names','psd']].values.tolist())
            lowpsd=get_value(column_names[i],lowdata[['names','psd']].values.tolist())
            athletepsd=get_value(column_names[i],athletedata[['names','psd']].values.tolist())
            highpsd_all[i].append(highpsd)
            lowpsd_all[i].append(lowpsd)
            athletepsd_all[i].append(athletepsd)
    
    
    band_num=5
    x=[i+1 for i in range(3)]
    for i in range(band_num):
        curband_highpsd=np.array(highpsd_all[i]).T
        curband_lowpsd=np.array(lowpsd_all[i]).T
        curband_athletepsd=np.array(athletepsd_all[i]).T
        curchannels=column_names[i]

        print(curband_highpsd)
        for j in range(len(curchannels)):
            #plt.plot(x,curband_highpsd[j],label="high_"+curchannels[j]+"_"+bands[i][2])
            #plt.plot(x,curband_lowpsd[j],label="low_"+curchannels[j]+"_"+bands[i][2])
            #plt.plot(x,curband_athletepsd[j],label="athlete_"+curchannels[j]+"_"+bands[i][2])
            plt.plot(x,curband_highpsd[j],label="high")
            plt.plot(x,curband_lowpsd[j],label="low")
            plt.plot(x,curband_athletepsd[j],label="athlete")
            plt.legend()
            plt.title(curchannels[j]+"_"+bands[i][2])
            plt.savefig(os.path.join(save_path,curchannels[j]+"_"+bands[i][2]+".jpg"))
            plt.close()


if __name__=="__main__":
    compare_3s_12epoch_psd()
    #compare_3s_before_shot()
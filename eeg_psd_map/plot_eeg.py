import mne
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch

#修改路径为去除眼电后.fif目录文件
path_config=r""
path=os.path.join(path_config,"*.fif")
save_path=os.path.join(path_config,"outcome")

#需要提取的标签名称
markernames=["s9003","s9004","s9005","s9006","s9007","s9008","s9009"]
#功率谱脑地形图的上下幅值
limit=[(0,0.9),(0,0.2),(0,0.9),(0,0.6),(-0.2,0.2)]

def init():
    if not os.path.exists(save_path):
        os.makedirs(save_path)


def read_fif():
    global path,save_path,markernames,limit
    files=glob(path)
    for f in files:
        data=mne.io.read_raw(f)
        print(f)
        events,event_id=mne.events_from_annotations(data)
        #print(data.info)
        epochs=mne.Epochs(data,events,event_id,tmin=-3,tmax=3,event_repeated='drop',preload=True)
        print(f[:-4])
        save_path_person=os.path.join(save_path,re.findall(r".*\\(.*)",f[:-4])[0])
        if not os.path.exists(save_path_person):
            os.makedirs(save_path_person)
        epochs=epochs[markernames]
        bands = [[(0, 4, 'Delta')], [(4, 8, 'Theta')], [(8, 12, 'Alpha')],
         [(12, 30, 'Beta')], [(30, 45, 'Gamma')]]

        for markername in markernames:
            for count,band in enumerate(bands):
                print(markername)
                fig=epochs[markername].plot_psd_topomap(bands=band,normalize=True,vlim=limit[count],show=False,cmap="Reds")
                cur_path=os.path.join(save_path_person,markername+"_"+band[0][2]+".jpg")
                fig.savefig(cur_path)

init()

if __name__=="__main__":
    read_fif()

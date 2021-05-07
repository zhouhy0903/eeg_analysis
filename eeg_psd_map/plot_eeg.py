import mne
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
from spectrum import Periodogram, TimeSeries

#修改路径为去除眼电后.fif目录文件
path_config=r"D:\data\jxy\data"
path=os.path.join(path_config,"*.fif")
save_path=os.path.join(path_config,"outcome")

#需要提取的标签名称
markernames=["s9003","s9004","s9005","s9006","s9007","s9008","s9009"]
channelnames=["Fz","F3","F4","P3"]
#功率谱脑地形图的上下幅值
limit=[(0,0.9),(0,0.2),(0,0.9),(0,0.6),(-0.2,0.2)]

def init():
    if not os.path.exists(save_path):
        os.makedirs(save_path)


def save_psd():
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


def save_channel_psd():
    global path,save_path,markernames,channelnames
    files=glob(path)
    for f in files:
        
        data=mne.io.read_raw(f)
        print("Reading from {}".format(f))
        events,event_id=mne.events_from_annotations(data)
        epochs=mne.Epochs(data,events,event_id,tmin=-3,tmax=3,event_repeated='drop',preload=True)[markernames]


        for channelname in channelnames:
            plot_channel_psd(epochs["s9003"],channelname)
        break


def plot_channel_psd(data_eeg,channel):
    def get_band(data,chname):
        p=Periodogram(data,sampling=500)
        p.run()
        frepsd=pd.DataFrame()
        frepsd["freq"]=p.frequencies()
        frepsd["psd"]=10*np.log10(p.psd)
        frepsd.drop(frepsd[frepsd["freq"]>=45].index,inplace=True)

        fig = plt.figure(1)
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.plot(frepsd["freq"].tolist(),frepsd["psd"].tolist(),color="blue")
        plt.xlabel("Frequency/Hz")
        plt.ylabel("PSD $\mu V^2/Hz (dB)$")
        plt.title(chname)
        plt.show()
    get_band(data_eeg.to_data_frame()[channel].tolist(),channel)

init()

if __name__=="__main__":
    #save_psd()
    save_channel_psd()

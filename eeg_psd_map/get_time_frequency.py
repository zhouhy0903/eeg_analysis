import mne
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch

path=r"D:\data\jxy\new\outcome\eeg_removeeye\*.fif"
save_path=r"D:\data\jxy\new\outcome\eeg_removeeye\test"

def init():
    if not os.path.exists(save_path):
        os.makedirs(save_path)


def read_fif():
    global path,save_path
    files=glob(path)
    for f in files:
        data=mne.io.read_raw(f)
        print(f)
        events,event_id=mne.events_from_annotations(data)
        #channel_names=data.channel_names
        print(data.info)
        epochs=mne.Epochs(data,events,event_id,tmin=-3,tmax=3,event_repeated='drop',preload=True)
        markernames=["s9003","s9004","s9005","s9006","s9007","s9008","s9009"]
        print(f[:-4])
        save_path_person=os.path.join(save_path,re.findall(r".*\\(.*)",f[:-4])[0])
        if not os.path.exists(save_path_person):
            os.makedirs(save_path_person)
        epochs=epochs[markernames]
        for markername in markernames:
            print(markername)
            fig=epochs[markername].plot_psd_topomap(normalize=True,show=False)
            cur_path=os.path.join(save_path_person,markername+".jpg")
            fig.savefig(cur_path)
        """
        freqs = np.logspace(*np.log10([0.5, 45]), num=8)
        n_cycles = freqs / 2.  # different number of cycle per frequency

        power, itc = tfr_morlet(epochs["s9005"], freqs=freqs, n_cycles=n_cycles, use_fft=True,
            return_itc=True, decim=3, n_jobs=1)
        
        channel=0
        for channel in range(31):
            power.plot([channel],baseline=(-3,3),mode='logratio')
            cur_path=os.path.join(save_path,str(channel))
            #plt.savefig(cur_path)
            break

        break
    """

init()
read_fif()

#if __name__=="__main__":
 #   read_fif()

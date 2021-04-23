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


files=[]
def init():
    global files
    cf=configparser.ConfigParser()
    cf.read("eeg_path.conf")
    path_names=cf.options("path")
    path_eeg=cf.get("path","path_eeg")
    print(path_eeg)
    files=glob(os.path.join(path_eeg,"*.fif"))
    

def preprocess():
    global files
    count=0

    for i,f in enumerate(files):
        try:
            print("Reading eeg from {}".format(f))
            data=mne.io.read_raw_fif(f)
            data=data.set_montage("standard_1020")
            events,event_id=mne.events_from_annotations(data)
            epochs=mne.Epochs(data,events,event_id,tmin=-5,tmax=0,event_repeated='drop',preload=True)
            first,second=get_score(i+1)
            print(len(first),len(second))
            rest_eeg=epochs["s1001"][0]
            total_shot_num=len(first)+len(second)
            first_shoot_eeg=epochs["s1002"][-total_shot_num:-len(second)]
            second_shoot_eeg=epochs["s1002"][-len(second):]
            f=0
            s=0

            for j in range(len(epochs)-1,1,-1):
                if epochs.events[j][2]==4:
                    s+=1
                else:
                    if s>0: break
            for k in range(j,1,-1):
                if epochs.events[k][2]==6: continue 
                else: break
            for j in range(k,1,-1):
                if epochs.events[j][2]==4:
                    f+=1
                else:
                    if f>0: break
            #if f!=len(first): continue
            freqs = np.logspace(*np.log10([0.5, 45]), num=8)
            n_cycles = freqs / 2.  # different number of cycle per frequency
            print("test") 

            power, itc = tfr_morlet(rest_eeg, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=3, n_jobs=1)
            channel=0
            fig, axis = plt.subplots(1, 2, figsize=(7, 4))
            for channel in range(31):
                power.plot([channel],baseline=(-5,0),mode='logratio')
                plt.show()
            #power.plot_joint(mode='mean', tmin=-2, tmax=0,
            #     timefreqs=[(.5, 10), (1.3, 8)])
            break
        except Exception as e:
            traceback.print_exc()
            break
            pass
    pass

init()

if __name__=="__main__":
   preprocess() 
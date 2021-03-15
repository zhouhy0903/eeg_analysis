import mne
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm

from mne.preprocessing import ICA
path,eegn_path,eegp_path,save_path="","","",""

def init_path():
    global path,eegn_path,eegp_path,save_path
    path="D:\data\eegemotion"
    eegn_path=os.path.join(path,"eeg-N")
    eegp_path=os.path.join(path,"eeg-P")
    save_path=os.path.join(path,"preprocess")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

class eeg():
    file_paths=[]
    def __init__(self,paths):
        print("get eeg file path")
        for _,file_dirs,_ in os.walk(paths):
            for file_dir in file_dirs:
                first_path=os.path.join(paths,file_dir)
                for _,file_times,_ in os.walk(first_path):
                    for file_time in file_times:
                        final_path=os.path.join(first_path,file_time)
                        files=os.listdir(final_path)
                        final_path=os.path.join(final_path,files[1])
                        self.file_paths.append(final_path)
        print("path saved")

    def get_eeg(self,num):   
        if num>len(self.file_paths): return
        print("reading {}".format(self.file_paths[num-1]))
        raw=mne.io.read_raw(self.file_paths[num-1],preload=True)
        return raw

    def preprocess(self,num):
        raw=self.get_eeg(num)
        raw=raw.filter(0.5,20)
        
        events,event_id=mne.events_from_annotations(raw)
        epochs=mne.Epochs(raw,events,event_id,tmin=-5,tmax=0,event_repeated='drop')
        print(events)
        epochs["s1002"][-40:].plot(n_epochs=3) 
        plt.show()
        shoot_eeg=epochs["s1002"][-40:]
        ica=ICA(n_components=15,random_state=90)
        
        ica.fit(shoot_eeg)
        ica.plot_sources(shoot_eeg,show_scrollbars=True)
        plt.show()


if __name__=='__main__':
    init_path()
    eegdata=eeg(eegn_path)
    eegdata.preprocess(1) 

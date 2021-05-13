from score import get_score
from state import get_state
import matplotlib.pyplot as plt
import traceback
import mne
import numpy as np
import pandas as pd
from spectrum import Periodogram, TimeSeries
from eegprocess import get_raw_eeg,get_epoch_eeg
def draw_score_distribution():
    score=[]
    for i in range(1,60):
        try:
            score.extend(get_score(i)[0])
        except Exception as e:
            pass
    print(score)
    plt.hist(score,width=0.8,color="orange")
    plt.title("shot score distribution")
    plt.xlabel("shot score")
    plt.ylabel("num")
    plt.show()

def discover_psd_difference():
    highscore_epoch,lowscore_epoch=None,None
    highscorenum,lowscorenum=0,0
    for i in range(1,60):
        try:
            data_score=get_score(i)[0]
            data_state=get_state(i,3)
            data_eeg=get_epoch_eeg(i).drop(["condition"],axis=1)
            data_raw_eeg=get_raw_eeg(i)
            if not (len(data_score)==len(data_eeg["epoch"].value_counts()) and len(data_score)==len(data_state[data_state["markerText"]=="ShotOps"])): continue
            print("yes")
            
            events,event_id=mne.events_from_annotations(data_raw_eeg)
            #print(events)
            epochs=mne.Epochs(data_raw_eeg,events,event_id,tmin=-3,tmax=0,event_repeated='drop',preload=True)
            first,second=get_score(i+1)
            rest_eeg=epochs["s1001"][0]
            total_shot_num=len(first)+len(second)
            first_shoot_eeg=epochs["s1002"][-total_shot_num:-len(second)]
            for j in range(len(data_score)):
                if data_score[j]>9:
                    if highscore_epoch==None:
                        highscore_epoch=first_shoot_eeg["s1002"][j]
                    else:
                        highscore_epoch=mne.concatenate_epochs([highscore_epoch,first_shoot_eeg["s1002"][j]])
                    highscorenum+=1

                if data_score[j]<8:
                    if lowscore_epoch==None:
                        lowscore_epoch=first_shoot_eeg["s1002"][j]
                    else:
                        lowscore_epoch=mne.concatenate_epochs([lowscore_epoch,first_shoot_eeg["s1002"][j]])
                    lowscorenum+=1
            print(highscorenum)
            print(lowscorenum)
            #fig1=highscore_epoch.plot_psd_topomap(show=False,normalize=True)
            #fig2=lowscore_epoch.plot_psd_topomap(show=False,normalize=True)
            #fig1.savefig(r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\high_score_3s_test\high_score{}.jpg".format(i))
            #fig2.savefig(r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\low_score_3s_test\low_score{}.jpg".format(i))
        except Exception as e:
            traceback.print_exc()
    fig1=highscore_epoch.plot_psd_topomap(show=False,cmap="RdBu_r",normalize=True)
    fig2=lowscore_epoch.plot_psd_topomap(show=False,cmap="RdBu_r",normalize=True)
    fig1.savefig(r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\high_score.jpg")
    fig2.savefig(r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\low_score.jpg")
    print(highscorenum)
    print(lowscorenum)


def discover_event_difference():
    highscore_epoch,lowscore_epoch=None,None
    highscorenum,lowscorenum=0,0
    for i in range(1,60):
        try:
            data_score=get_score(i)[0]
            data_state=get_state(i,3)
            data_eeg=get_epoch_eeg(i).drop(["condition"],axis=1)
            data_raw_eeg=get_raw_eeg(i)
            if not (len(data_score)==len(data_eeg["epoch"].value_counts()) and len(data_score)==len(data_state[data_state["markerText"]=="ShotOps"])): continue
            print("yes")
            
            events,event_id=mne.events_from_annotations(data_raw_eeg)
            #print(events)
            epochs=mne.Epochs(data_raw_eeg,events,event_id,tmin=-3,tmax=0,event_repeated='drop',preload=True)
            first,second=get_score(i+1)
            rest_eeg=epochs["s1001"][0]
            total_shot_num=len(first)+len(second)
            first_shoot_eeg=epochs["s1002"][-total_shot_num:-len(second)]
            for j in range(len(data_score)):
                if data_score[j]>9:
                    if highscore_epoch==None:
                        highscore_epoch=first_shoot_eeg["s1002"][j]
                    else:
                        highscore_epoch=mne.concatenate_epochs([highscore_epoch,first_shoot_eeg["s1002"][j]])
                    highscorenum+=1

                if data_score[j]<8:
                    if lowscore_epoch==None:
                        lowscore_epoch=first_shoot_eeg["s1002"][j]
                    else:
                        lowscore_epoch=mne.concatenate_epochs([lowscore_epoch,first_shoot_eeg["s1002"][j]])
                    lowscorenum+=1
            print(highscorenum)
            print(lowscorenum)
            #fig1=highscore_epoch.plot_psd_topomap(show=False,normalize=True)
            #fig2=lowscore_epoch.plot_psd_topomap(show=False,normalize=True)
            #fig1.savefig(r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\high_score_3s_test\high_score{}.jpg".format(i))
            #fig2.savefig(r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\low_score_3s_test\low_score{}.jpg".format(i))
        except Exception as e:
            traceback.print_exc()

    freq_low=[0,4,8,12,30]
    freq_high=[4,8,12,30,45]
    for j in range(len(freq_low)):
        highscore_epoch.load_data().filter(l_freq=freq_low[j],h_freq=freq_high[j])
        highscoredata=highscore_epoch.get_data()
        corr_matrix=mne.connectivity.envelope_correlation(highscoredata,combine="mean")
        plt.imshow(corr_matrix)
        plt.colorbar()
        plt.title("highscore_({}Hz-{}Hz)".format(freq_low[j],freq_high[j]))
        plt.savefig(r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\diff\corr\high_score\high_score({}Hz-{}Hz).jpg".format(freq_low[j],freq_high[j]))
        plt.close()

        lowscore_epoch.load_data().filter(l_freq=freq_low[j],h_freq=freq_high[j])
        lowscoredata=lowscore_epoch.get_data()
        corr_matrix=mne.connectivity.envelope_correlation(lowscoredata,combine="mean")
        plt.imshow(corr_matrix)
        plt.colorbar()
        plt.title("lowscore_({}Hz-{}Hz)".format(freq_low[j],freq_high[j]))
        plt.savefig(r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\diff\corr\low_score\low_score({}Hz-{}Hz).jpg".format(freq_low[j],freq_high[j]))
        plt.close()
    #fig1.savefig(r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\diff\event\high_score.jpg")
    #fig2.savefig(r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\diff\event\low_score.jpg")
    print(highscorenum)
    print(lowscorenum)

def discover_channel_psd_difference():
    highscore_epoch,lowscore_epoch=None,None
    highscorenum,lowscorenum=0,0
    for i in range(1,60):
        try:
            data_score=get_score(i)[0]
            data_state=get_state(i,3)
            data_eeg=get_epoch_eeg(i).drop(["condition"],axis=1)
            data_raw_eeg=get_raw_eeg(i)
            if not (len(data_score)==len(data_eeg["epoch"].value_counts()) and len(data_score)==len(data_state[data_state["markerText"]=="ShotOps"])): continue
            print("yes")
            events,event_id=mne.events_from_annotations(data_raw_eeg)
            epochs=mne.Epochs(data_raw_eeg,events,event_id,tmin=-3,tmax=0,event_repeated='drop',preload=True)
            first,second=get_score(i+1)
            rest_eeg=epochs["s1001"][0]
            total_shot_num=len(first)+len(second)
            first_shoot_eeg=epochs["s1002"][-total_shot_num:-len(second)]
            for j in range(len(data_score)):
                if data_score[j]>9:
                    if highscore_epoch==None:
                        highscore_epoch=first_shoot_eeg["s1002"][j]
                    else:
                        highscore_epoch=mne.concatenate_epochs([highscore_epoch,first_shoot_eeg["s1002"][j]])
                    highscorenum+=1

                if data_score[j]<8:
                    if lowscore_epoch==None:
                        lowscore_epoch=first_shoot_eeg["s1002"][j]
                    else:
                        lowscore_epoch=mne.concatenate_epochs([lowscore_epoch,first_shoot_eeg["s1002"][j]])
                    lowscorenum+=1
            print(highscorenum)
            print(lowscorenum)
        except Exception as e:
            traceback.print_exc()
    channelnames=["Fz","F3","F4","P3","Pz","P4","O1","O2","POz"]
    print("test")
    for channelname in channelnames:
        plot_channel_psd(highscore_epoch,channelname)

    for channelname in channelnames:
        plot_channel_psd(lowscore_epoch,channelname)
        
def plot_channel_psd(data_eeg,channel):
    def get_band(data,chname):
        p=Periodogram(data,sampling=1000)
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


#discover_psd_difference()
#discover_event_difference()
discover_channel_psd_difference()
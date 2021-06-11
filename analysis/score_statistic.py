from typing import NoReturn
from score import get_score
from state import get_state
import matplotlib.pyplot as plt
import traceback
import mne
import numpy as np
import pandas as pd
from spectrum import Periodogram, TimeSeries
from eegprocess import get_raw_eeg,get_epoch_eeg
import numpy as np
import os

def draw_score_distribution():
    score=[]
    for i in range(1,60):
        try:
            score.extend(get_score(i)[0])
        except Exception as e:
            pass
    print(score)
    plt.hist(score,width=0.8,color="orange")
    plt.title("射击成绩分布")
    plt.xlabel("射击成绩/分")
    plt.ylabel("个数")
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus']=False
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
            data_raw_eeg.plot()
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
            fig1=highscore_epoch.plot_psd_topomap(show=False,normalize=False,cmap="RdBu_r")
            #fig2=lowscore_epoch.plot_psd_topomap(show=False,normalize=False,cmap="RdBu_r")
            fig1.savefig(r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\high_low1\high_score{}.jpg".format(i))
            #fig2.savefig(r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\high_low1\low_score{}.jpg".format(i))
            #break
        except Exception as e:
            traceback.print_exc()
            #break
    fig1=highscore_epoch.plot_psd_topomap(show=False,cmap="RdBu_r",normalize=False)
    fig2=lowscore_epoch.plot_psd_topomap(show=False,cmap="RdBu_r",normalize=False)
    fig1.savefig(r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\high_low1\high_score2.jpg")
    fig2.savefig(r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\high_low1\low_score2.jpg")
    print(highscorenum)
    print(lowscorenum)

def get_psd_data(bands,data_epoch,picks):
    psds,freqs=mne.time_frequency.psd_multitaper(inst=data_epoch,picks=picks)
    psds = np.mean(psds, axis=0)
    psds /= psds.sum(axis=-1, keepdims=True)
        
    """
    scaling=1000000
    psds *= scaling * scaling
    np.log10(np.maximum(psds, np.finfo(float).tiny), out=psds)
    psds *= 10
    """
    normalize=True
    agg_fun=None
    dB=True

    if agg_fun is None:
        agg_fun = np.sum if normalize else np.mean
    
    if bands is None:
        bands = [(0, 4, 'Delta (0-4 Hz)'), (4, 8, 'Theta (4-8 Hz)'),
                (8, 12, 'Alpha (8-12 Hz)'), (12, 30, 'Beta (12-30 Hz)'),
                (30, 45, 'Gamma (30-45 Hz)')]
    data_list=[]
    
    for i,(fmin, fmax, title) in enumerate(bands):
        freq_mask = (fmin < freqs) & (freqs < fmax)
        if freq_mask.sum() == 0:
            raise RuntimeError('No frequencies in band "%s" (%s, %s)'
                            % (title, fmin, fmax))
        data = agg_fun(psds[:, freq_mask], axis=1)
        if dB and not normalize:
            data = 10 * np.log10(data)
        data_list.append(data)
        """
        print(column_names,data)
        plt.bar(column_names,data,color="red")
        plt.title("Normalized PSD {}".format(title))
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(savepath,"psdfreqs_{}".format(title)))
        """
    return data_list


def discover_psd_difference_time():
    #start=[-3+i*0.2 for i in range(15)]
    #end=[-3+(i+1)*0.2 for i in range(15)]
    #start=[-3+i*1 for i in range(3)]
    #end=[-3+(i+1)*1 for i in range(3)]
    
    start=[-3]
    end=[0]
    column_names=['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2']
    #column_names=[['F3', 'Fz', 'F4'], 
                #   ['F3', 'Fz', 'F4'], 
                #   ['F3', 'Fz', 'F4','P7', 'P3', 'P4', 'P8','O1','O2','M1'], 
                #   ['F3', 'Fz', 'F4','P7', 'P3', 'P4', 'P8','O1','O2','M1'], 
                #   ['F3', 'Fz', 'F4','P7', 'P3', 'P4', 'P8','O1','O2','M1']]
    
    for timei in range(len(start)):
        highscore_epoch,lowscore_epoch=None,None
        highscore_score,lowscore_score=[],[]
        highscorenum,lowscorenum=0,0
        print("Geting eeg from {} to {} before shot".format(start[timei],end[timei]))
        savepath=r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\csv0_3\{}_{}".format(start[timei],end[timei])
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        for i in range(1,60):
            try:
                data_score=get_score(i)[0]
                data_state=get_state(i,3)
                data_eeg=get_epoch_eeg(i).drop(["condition"],axis=1)
                data_raw_eeg=get_raw_eeg(i)

                if not column_names:
                    column_names=data_raw_eeg.info.ch_names
                if not (len(data_score)==len(data_eeg["epoch"].value_counts()) and len(data_score)==len(data_state)): continue
                events,event_id=mne.events_from_annotations(data_raw_eeg)
                epochs=mne.Epochs(data_raw_eeg,events,event_id,tmin=start[timei],tmax=end[timei],baseline=(start[timei],end[timei]),event_repeated='drop',preload=True)
                first,second=get_score(i+1)
                rest_eeg=epochs["s1001"][0]
                total_shot_num=len(first)+len(second)
                first_shoot_eeg=epochs["s1002"][-total_shot_num:-len(second)]
                for j in range(len(data_score)):
                    if data_score[j]>9:
                        if highscore_epoch==None:
                            highscore_epoch=first_shoot_eeg["s1002"][j]
                            highscore_score.append(data_score[j])
                        else:
                            highscore_epoch=mne.concatenate_epochs([highscore_epoch,first_shoot_eeg["s1002"][j]])
                            highscore_score.append(data_score[j])
                        highscorenum+=1
                    if data_score[j]<8:
                        if lowscore_epoch==None:
                            lowscore_epoch=first_shoot_eeg["s1002"][j]
                            lowscore_score.append(data_score[j])
                        else:
                            lowscore_epoch=mne.concatenate_epochs([lowscore_epoch,first_shoot_eeg["s1002"][j]])
                            lowscore_score.append(data_score[j])
                        lowscorenum+=1
                

                print(highscorenum)
                print(lowscorenum)
                """
                if i==1:
                    fig1=highscore_epoch.plot_psd_topomap(show=False,cmap="RdBu_r",normalize=True)
                    fig2=lowscore_epoch.plot_psd_topomap(show=False,cmap="RdBu_r",normalize=True)
                    fig1.savefig(os.path.join(savepath,"1_high.jpg"))
                    fig2.savefig(os.path.join(savepath,"1_low.jpg"))
                """
                #break
            except Exception as e:
                traceback.print_exc()
        fig1=highscore_epoch.plot_psd_topomap(show=False,cmap="RdBu_r",normalize=True)
        fig2=lowscore_epoch.plot_psd_topomap(show=False,cmap="RdBu_r",normalize=True)
        fig1.savefig(os.path.join(savepath,"all_high.jpg"))
        fig2.savefig(os.path.join(savepath,"all_low.jpg"))
        
        print(column_names)
        print(highscorenum)
        bands = [(0, 4, '1Delta (0-4 Hz)'), (4, 8, '2Theta (4-8 Hz)'),
                (8, 12, '3Alpha (8-12 Hz)'), (12, 30, '4Beta (12-30 Hz)'),
                (30, 45, '5Gamma (30-45 Hz)')]

        #highscore_psd_data=get_psd_data(bands=bands,data_epoch=highscore_epoch,picks=column_names)
        #lowscore_psd_data=get_psd_data(bands=bands,data_epoch=lowscore_epoch,picks=column_names)
        for num in range(len(highscore_epoch)):
            for i,band in enumerate(bands):
                highscore_psd_data=get_psd_data(bands=[band],data_epoch=highscore_epoch[num],picks=column_names)
                df=pd.DataFrame()
                df["names"]=column_names
                df["psd"]=highscore_psd_data[0]
                cur_path=os.path.join(savepath,str(num)+"_high_"+str(highscore_score[num]))
                if not os.path.exists(cur_path):
                    os.makedirs(cur_path)
                df.to_csv(os.path.join(cur_path,"highscore{}.csv".format(i)))

            plt.bar(column_names[i],highscore_psd_data[0],color="red")
            plt.title("高水平射击 {} 各导联功率谱密度".format(band[2]))
            plt.xticks(rotation=90)
            plt.rcParams['font.family'] = ['sans-serif']
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus']=False
            plt.savefig(os.path.join(savepath,"psdfreqs_highscore_{}".format(band[2])))
            plt.close()

        
        for num in range(len(lowscore_epoch)):
            for i,band in enumerate(bands):
                lowscore_psd_data=get_psd_data(bands=[band],data_epoch=lowscore_epoch[num],picks=column_names)
                df=pd.DataFrame()
                df["names"]=column_names
                df["psd"]=lowscore_psd_data[0]
                cur_path=os.path.join(savepath,str(num)+"_low_"+str(lowscore_score[num]))
                if not os.path.exists(cur_path):
                    os.makedirs(cur_path)
                df.to_csv(os.path.join(cur_path,"lowscore{}.csv".format(i)))
            
            plt.bar(column_names[i],lowscore_psd_data[0],color="green")
            plt.title("低水平射击 {} 各导联功率谱密度".format(band[2]))
            plt.xticks(rotation=90)
            plt.rcParams['font.family'] = ['sans-serif']
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus']=False
            plt.savefig(os.path.join(savepath,"psdfreqs_lowscore_{}".format(band[2])))
            plt.close()


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

#draw_score_distribution()
discover_psd_difference_time()
#discover_event_difference()
#discover_channel_psd_difference()
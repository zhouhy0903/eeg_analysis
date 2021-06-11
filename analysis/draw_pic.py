import matplotlib.pyplot as plt
from eegprocess import get_raw_eeg,get_epoch_eeg
import mne
import seaborn as sns
import pandas as pd
import seaborn as sns




def draw_eeg_split():
    start=[i for i in range(3)]
    end=[-3+(i+1) for i in range(3)]

    epoch_all=None
    data_raw_eeg=get_raw_eeg(1)
    events,event_id=mne.events_from_annotations(data_raw_eeg)
    epochs=mne.Epochs(data_raw_eeg,events,event_id,tmin=-3,tmax=0,event_repeated='drop',preload=True)
        
    df=epochs.to_data_frame()
    df["Fz"].plot()
    for i in range(3):
        plt.vlines(start[i]*100000,-110,110,color="green")

    plt.xlabel(r"时间t/ $\times 10^{-5}$ s")
    plt.title("射击前3s脑电信号分段")
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.show()

def draw_ttest_outcome():
    data=pd.read_csv("ttest.csv",index_col=0)
    print(data)
    sns.clustermap(data, cmap="coolwarm",row_cluster=False,col_cluster=False)
    plt.show()
    #plt.savefig("ttest.jpg")

def analyse_knn():
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
    data=pd.read_csv("outcome_aimtrack.csv")
    print(data)
    data.plot(kind='bar',cmap="cividis", legend=True)
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.title("高低水平分类结果")
    plt.show()
    print(data["precision"].mean())
    print(data["recall"].mean())
    print(data["f1-score"].mean())
    
    print(data)

if __name__=="__main__":
    # analyse_knn()
    
    draw_eeg_split()
    #draw_ttest_outcome()
from glob import glob
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

bands = [(0, 4, 'Delta (0-4 Hz)'), (4, 8, 'Theta (4-8 Hz)'),
         (8, 12, 'Alpha (8-12 Hz)'), (12, 30, 'Beta (12-30 Hz)'),
         (30, 45, 'Gamma (30-45 Hz)')]
band_name=[]

for band in bands:
    band_name.append(band[2])
print(band_name)
channel_names=None
def get_lowhighdata():
    global channel_names
    path=r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\csv_t_3\-0.5_0\*"
    files=glob(path)
    highpsd,lowpsd=[],[]
    allpsd=[]
    score=[]
    for file in tqdm(files):
        score.append(float(re.findall(r".*\\.*_.*_(.*)",file)[0]))
        curpath=os.path.join(file,"*.csv")
        bandfiles=glob(curpath)
        psd=[]
        for bandf in bandfiles:
            data=pd.read_csv(bandf,index_col=0)
            if not channel_names:
                channel_names=data["names"].tolist()
            psd.append(data["psd"].tolist())
        hol=re.findall(r".*\\.*_(.*)_.*",file)[0]
        if hol=="high":
            highpsd.append(psd)
        if hol=="low":
            lowpsd.append(psd)
        allpsd.append(psd)
        


    highpsd_np=np.array(highpsd)
    lowpsd_np=np.array(lowpsd)
    print(highpsd_np.shape)
    print(lowpsd_np.shape)
    return highpsd,lowpsd,allpsd,score

def get_channel_band(pick,psd):
    global channel_names
    keys=channel_names
    values=[i for i in range(len(channel_names))]
    channel=dict(zip(keys,values))
    if pick not in channel_names:
        raise Exception("Channel not found!")

    psd_np=np.array(psd)
    psd_channel=psd_np[:,:,channel[pick]]
    return psd_channel

def ttest_psd():
    highpsd,lowpsd,_,_=get_lowhighdata()
    bandp_all=[]
    for channel_name in channel_names:
        highpsd_channel=get_channel_band(channel_name,highpsd)
        lowpsd_channel=get_channel_band(channel_name,lowpsd)
        bandp_channel=[] 
        for i,band in enumerate(bands):
            highpsd_channel_band=highpsd_channel[:,i]
            lowpsd_channel_band=lowpsd_channel[:,i]
            result=stats.ttest_rel(a=highpsd_channel_band,b=lowpsd_channel_band)
            print(result)
            print(result[1])
            bandp_channel.append(result[1])
        bandp_all.append(bandp_channel)
    df=pd.DataFrame(bandp_all,columns=band_name,index=channel_names)
    print(df)
    df.to_csv("ttest.csv")


def psd_regress():
    #global channel_names

    _,_,all_psd,score=get_lowhighdata()
    #area_channels=[["F3","Fz","F4"],["P3","Pz","P4"],["O1","O2"]]
    area_channels=[["Fp1","Fpz","Fp2"],["F7","F3","Fz","F4","F8"],["FC5","FC1","FC2","FC6"],["M1","M2"],["P3","Pz","P4","P7","P8"],["O1","O2"]]
    #area_channels=np.array(channel_names).reshape(31,1).tolist()
    print(area_channels)
    area_name=["Fp","F","FC","M","P","O"]
    area_mean_all=[]
    regress_data=pd.DataFrame()
    for idx,area_channel in enumerate(area_channels):
        average=0
        area_merge=[]
        for channel in area_channel:
            singlepsd_channel=get_channel_band(channel,all_psd)
            singlepsd_channel=np.expand_dims(singlepsd_channel, 2)
            if len(area_merge)==0:
                area_merge=singlepsd_channel
            else:
                area_merge=np.concatenate((area_merge,singlepsd_channel),axis=2)

        area_mean=np.mean(area_merge,axis=2)
        print(area_mean.shape)
        area_mean_all.append(area_mean)
        n1="delta_{}".format(area_name[idx])
        n2="theta_{}".format(area_name[idx])
        n3="alpha_{}".format(area_name[idx])
        n4="beta_{}".format(area_name[idx])
        n5="gamma_{}".format(area_name[idx])
        #print(area_mean[:,0].tolist())
        #print(area_mean[:,1].tolist())
        regress_data[n1]=area_mean[:,0].tolist()
        regress_data[n2]=area_mean[:,1].tolist()
        regress_data[n3]=area_mean[:,2].tolist()
        regress_data[n4]=area_mean[:,3].tolist()
        regress_data[n5]=area_mean[:,4].tolist()
    
    print(len(area_mean_all))
    print(score)
    regress_data["score"]=score
    print(regress_data)
    regress_data.to_csv("psd_regress.csv")
    regress_data.to_stata("psd_regress.dta")
    pass


from sklearn import tree
from sklearn.model_selection import KFold
from sklearn import svm

def fun(x):
    if x>10:
        return 0
    if x<6.5:
        return 1
    return 2

def decision_tree():

    data=pd.read_csv("psd_regress.csv",index_col=0)
    data["score_label"]=data["score"].apply(lambda x:fun(x))
    data.drop(data[data["score_label"]==2].index,inplace=True)
    x=np.array(data[data.columns[:6]])
    y=np.array(data[data.columns[-1]])
    xy=np.column_stack((x,y))
    kf = KFold(n_splits=10)
    accuracy=[]
    for train_index, test_index in kf.split(xy):
        train_XY = xy[train_index]
        test_XY= xy[test_index]
        train_X=train_XY[:,:-1]
        train_Y=train_XY[:,-1]
        test_X=test_XY[:,:-1]
        test_Y=test_XY[:,-1]
        clf=svm.SVC()
        clf=clf.fit(train_X,train_Y)
        y_pred=clf.predict(test_X)
        print("Accuracy: ",np.sum(y_pred==test_Y)/len(test_Y))
        accuracy.append(np.sum(y_pred==test_Y)/len(test_Y))
    print(np.array(accuracy).mean())
    plt.bar(range(len(accuracy)),accuracy,label="Accuracy",color="red")
    plt.show()
    #print(data)
    #data.to_csv("drop.csv",index=False)


def random_forest():
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_blobs
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.tree import DecisionTreeClassifier
    data=pd.read_csv("psd_regress.csv",index_col=0)
    data["score_label"]=data["score"].apply(lambda x:fun(x))
    data.drop(data[data["score_label"]==2].index,inplace=True)
    x=np.array(data[data.columns[:6]])
    y=np.array(data[data.columns[-1]])
    xy=np.column_stack((x,y))

    #clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
    clf= RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
    clf=clf.fit(x,y)
    scores = cross_val_score(clf, x, y, cv=10,scoring="accuracy")
    print("Scores: ",scores.mean())

if __name__=="__main__":
    #psd_regress()
    #decision_tree()
    random_forest()
from eegprocess import get_epoch_eeg,get_raw_eeg
from score import get_score
from state import get_state


from torch import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import time
import traceback
import mne

from spectrum import Periodogram, TimeSeries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import math

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.conv3=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1)
        self.mpool=nn.MaxPool2d(kernel_size=3)
        self.conv4=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)
        self.linear1=nn.Linear(384,100)
        self.linear2=nn.Linear(100,1)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=self.mpool(x)
        x=F.relu(self.conv4(x))
        x=x.view(-1,384)
        x=F.relu(self.linear1(x))
        x=F.relu(self.linear2(x))
        return x


class ds(Dataset):
    def __init__(self,x,y,n):
        self.n=n
        self.x=x
        self.y=y
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.n




def get_aeraeeg_psd(data_eeg):
    def get_band(data):
        p=Periodogram(data,sampling=1000)
        p.run()
        #p.plot(marker='o',c="red")
        #plt.xlim(0,40)
        #plt.ylim(-10,45)
        #plt.show()
        #plt.plot(p.frequencies(),10*np.log10(p.psd))
        #plt.show()
        frepsd=pd.DataFrame()
        frepsd["freq"]=p.frequencies()
        frepsd["psd"]=10*np.log10(p.psd)
        frepsd.drop(frepsd[frepsd["freq"]>=45].index,inplace=True)
        band=[[0,4],[4,8],[8,12],[12,30],[30,45]]
        band_num=0
        band_mean=[]
        for i in range(len(band)):
            band_data=frepsd[frepsd["freq"]<=band[i][1]][frepsd["freq"]>=band[i][0]]
            band_mean.append(band_data["psd"].mean())
        return band_mean
    
    channel_names=data_eeg.columns[2:]
    psd_feature=[]
    for i in channel_names:
        data=get_band(data_eeg[i].tolist())
        psd_feature.append(data)

    psd_feature_np=np.array(psd_feature).T
    """
    plt.imshow(psd_feature_np,cmap="PuBu")
    plt.colorbar(shrink=.92)
    plt.xticks(range(31),channel_names.tolist(),rotation=90)
    plt.yticks([0,1,2,3,4],["Delta","Theta","Alpha","Beta","Gamma"]) 
    plt.title("psd_mean~channel and band")
    plt.show()
    """
    return psd_feature_np.tolist()

def get_data():
    datax=[]
    datay=[]
    for i in tqdm(range(1,60)):
        try:
            data_score=get_score(i)[0]
            data_state=get_state(i,5)
            data_eeg=get_epoch_eeg(i).drop(["condition"],axis=1)
            if not (len(data_score)==len(data_eeg["epoch"].value_counts()) and len(data_score)==len(data_state[data_state["markerText"]=="ShotOps"])): continue

            print("yes")
            for shoot in range(len(data_score)):
                datax.append(get_aeraeeg_psd(data_eeg[data_eeg["epoch"]==i][-1000:]))
            datay.extend(data_score)

            """
            data=get_raw_eeg(i)
            events,event_id=mne.events_from_annotations(data)
            epochs=mne.Epochs(data,events,event_id,tmin=-5,tmax=0,event_repeated='drop',preload=True)
            first,second=get_score(i+1)
            rest_eeg=epochs["s1001"][0]
            total_shot_num=len(first)+len(second)
            first_shoot_eeg=epochs["s1002"][-total_shot_num:-len(second)]
            first_shoot_eeg["s1002"][0].plot_psd(fmin=2., fmax=40., average=True, spatial_colors=False,picks=["Fz"])
            plt.show()
            """
        except Exception as e:
            traceback.print_exc()
            pass
    return datax,datay

def train_test_model():
    pass

def show_outcome(x,y):
    xmean=np.mean(x)    
    ymean=np.mean(y)
    SSR,varx,vary=0,0,0
    for i in range(len(x)):
        diffx=x[i]-xmean
        diffy=y[i]-ymean
        SSR+=(diffx*diffy)
        varx+=diffx**2
        vary+=diffy**2
    
    SST=math.sqrt(varx*vary)
    R2=(SSR/SST)**2
    plt.scatter(x,y,label="R2={}".format(R2))
    plt.legend()
    plt.show()
    pass


def train_model():
    num_epochs=200
    learning_rate=0.000001
    batch_size=10
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model=ConvNet().to(device)


    """
    dataxy=get_data()
    with open("psd_score.txt","wb") as f:
        pickle.dump(dataxy,f)
    """
    dataxy=[]
    with open('psd_score.txt', 'rb') as file:
        dataxy = pickle.load(file)

    x=np.array(dataxy[0])
    y=np.array(dataxy[1])
    train_x_origin,test_x_origin,train_y_origin,test_y_origin=train_test_split(x,y)

    traindataset=ds(train_x_origin,train_y_origin,len(train_x_origin))
    testdataset=ds(test_x_origin,test_y_origin,len(test_x_origin))
    loss=nn.MSELoss()
    optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)


    train_loader=DataLoader(dataset=traindataset,batch_size=batch_size,shuffle=True,num_workers=0)
    test_loader=DataLoader(dataset=testdataset,batch_size=batch_size,shuffle=True,num_workers=0)

    train_loss=[]
    test_loss=[]

    for epoch in tqdm(range(num_epochs)):
        cur_train_loss=[]
        cur_test_loss=[]
        for i,(data,labels) in enumerate(train_loader):
            data=data.to(device).reshape(-1,1,5,31)
            data=data.type(torch.FloatTensor).to(device)
            labels=labels.to(device)
            y_pred=model(data).type(torch.FloatTensor).to(device)
            labels=labels.type(torch.FloatTensor).to(device)
            l=loss(y_pred,labels)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            cur_train_loss.append(l.item())
        train_loss.append(np.array(cur_train_loss).mean())
        print("train loss ",np.array(cur_train_loss).mean())

        for i,(test_x,test_y) in enumerate(test_loader):
            test_x=test_x.reshape(-1,1,5,31).type(torch.FloatTensor).to(device)
            test_y=test_y.to(device)
            out=model(test_x)
            l=loss(out,test_y)
            cur_test_loss.append(l.item())
        test_loss.append(np.array(cur_test_loss).mean())
        print("test loss ",np.array(cur_test_loss).mean())

    plt.plot(train_loss,label="train_loss")
    plt.plot(test_loss,label="test_loss")
    plt.legend()
    plt.show()
    predict=[]
    torch.save(model.state_dict(),"model1.pt")
    for i in range(len(test_x_origin)):
        predict.append(model(torch.from_numpy(test_x_origin[i]).reshape(-1,1,5,31).type(torch.FloatTensor).to(device)).item())
    return test_y_origin,predict

"""
dataxy=get_data()
with open("psd_score.txt","wb") as f:
    pickle.dump(dataxy,f)
"""
actual,predict=train_model()
show_outcome(actual,predict)
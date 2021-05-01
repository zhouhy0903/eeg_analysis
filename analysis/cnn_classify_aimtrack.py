from eegprocess import get_epoch_eeg,get_raw_eeg
from score import get_score
from state import get_state
from aimtrack import get_aimtrack

from torch import torch
from torch.utils.data import Dataset,DataLoader

from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torchvision

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
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.conv3=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1)
        self.mpool=nn.MaxPool2d(kernel_size=3)
        self.conv4=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)
        self.linear1=nn.Linear(2304,100)
        self.linear2=nn.Linear(100,2)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=self.mpool(x)
        x=F.relu(self.conv4(x))
        
        x=x.view(-1,2304)
        x=F.relu(self.linear1(x))
        x=F.relu(self.linear2(x))
        x = torch.sigmoid(x)
        return x


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        
        self.conv=nn.Sequential(
            nn.Conv2d(6,48,3,1,2),#in-channels,out-channels,kernel-size,stride
            nn.Conv2d(48,128,3,1,2),#stride=1,padding=2
            nn.ReLU(),
            nn.MaxPool2d(3,2),
        )
        self.fc=nn.Sequential(
            nn.Linear(8704,100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100,2)
        )
    def forward(self,img):
        feature=self.conv(img)
        feature=feature.view(-1,8704)
        output=self.fc(feature)
        return output

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

def get_aeraeeg_corr(data_eeg):
    return np.array(data_eeg[data_eeg.columns[2:]].corr()).tolist()


def get_data():
    datax=[]
    datay=[]
    high,low=0,0
    for i in tqdm(range(1,60)):
        if i==32: continue
        try:
            data_score=get_score(i)[0]
            data_state=get_state(i,3)
            data_eeg=get_epoch_eeg(i).drop(["condition"],axis=1)
            data_aimtrack=get_aimtrack(i)
            """
            extract aimtrack data
            """
            
            
            print(data_aimtrack)
            break

            if not (len(data_score)==len(data_eeg["epoch"].value_counts()) and len(data_score)==len(data_state[data_state["markerText"]=="ShotOps"])): continue
            print("yes")
            for j in range(len(data_score)):
                """
                if data_score[j]>9.5:
                    datay.append(0)
                    high+=1

                if data_score[j]<7.5:
                    datay.append(1)
                    low+=1
                """

            print(len(datay),len(datax))
            #if len(datay)!=len(datax):
            #    raise Exception("length not matched")
        except Exception as e:
            break
            traceback.print_exc()
            pass
    print(low,high)
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
    num_epochs=100
    learning_rate=0.00001
    batch_size=5
    channel_size=3
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model=ConvNet().to(device)
    #model=AlexNet().to(device)


    """
    dataxy=get_data()
    with open("psd_score.txt","wb") as f:
        pickle.dump(dataxy,f)
    """
    dataxy=[]
    with open("D:/code/code/eegemotion/git/model/aimtrack_classify/aimtrack_score_classify.txt","rb") as f:
        dataxy = pickle.load(f)
    x=np.array(dataxy[0])
    y=np.array(dataxy[1])
    train_x_origin,test_x_origin,train_y_origin,test_y_origin=train_test_split(x,y)

    traindataset=ds(train_x_origin,train_y_origin,len(train_x_origin))
    testdataset=ds(test_x_origin,test_y_origin,len(test_x_origin))
    loss=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)


    train_loader=DataLoader(dataset=traindataset,batch_size=batch_size,shuffle=True,num_workers=0)
    test_loader=DataLoader(dataset=testdataset,batch_size=batch_size,shuffle=True,num_workers=0)

    train_loss=[]
    test_loss=[]
    train_accuracy=[]
    test_accuracy=[]
    for epoch in tqdm(range(num_epochs)):
        cur_train_loss=[]
        cur_test_loss=[]
        right,total=0,0
        for i,(data,labels) in enumerate(train_loader):
            data=data.to(device).reshape(-1,channel_size,31,31)
            data=data.type(torch.FloatTensor).to(device)
            y_pred=model(data).to(device)
            labels=labels.to(device).long()
            l=loss(y_pred,labels)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            cur_train_loss.append(l.item())
            right+=(torch.argmax(y_pred,dim=1)==labels).sum().item()
            total+=batch_size
        
        train_loss.append(np.array(cur_train_loss).mean())
        train_accuracy.append(right/total)
        print("train loss ",np.array(cur_train_loss).mean()," accuracy: ",right/total)

        right,total=0,0
        for i,(test_x,test_y) in enumerate(test_loader):
            test_x=test_x.reshape(-1,channel_size,31,31).type(torch.FloatTensor).to(device)
            test_y=test_y.to(device).long()
            out=model(test_x)
            l=loss(out,test_y)
            cur_test_loss.append(l.item())
            
            right+=(torch.argmax(out,dim=1)==test_y).sum().item()
            total+=batch_size
        
        test_loss.append(np.array(cur_test_loss).mean())
        test_accuracy.append(right/total)
        print("test loss ",np.array(cur_test_loss).mean()," accuracy: ",right/total)

    plt.plot(train_loss,label="train_loss")
    plt.plot(test_loss,label="test_loss")
    plt.legend()
    plt.show()

    plt.plot(train_accuracy,label="train_accuracy")
    plt.plot(test_accuracy,label="test_accuracy")
    plt.legend()
    plt.show()
    
    torch.save(model.state_dict(),"D:/code/code/eegemotion/git/model/aimtrack_classify/model.pt")

def save_input_data():
    dataxy=get_data()
    with open("D:/code/code/eegemotion/git/model/aimtrack_classify/aimtrack_score_classify.txt","wb") as f:
        pickle.dump(dataxy,f)

def show_input_data():
    dataxy=[]
    with open("D:/code/code/eegemotion/git/model/aimtrack_classify/aimtrack_score_classify.txt","rb") as f:
        dataxy = pickle.load(f)
    x=np.array(dataxy[0])
    y=np.array(dataxy[1])
    print(x.shape,y.shape)
    total=0
    sum=0
    for i in range(len(y)):
        sum+=y[i]
        total+=1
    print(sum,total)
#show_input_data()
#save_input_data()
get_data()
#train_model()
#show_outcome(actual,predict)

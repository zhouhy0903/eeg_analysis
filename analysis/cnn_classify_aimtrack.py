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

class FCN_model(nn.Module):
    def __init__(self,NumClassesOut,N_time,N_Features,N_LSTM_Out=128,
            N_LSTM_layers = 1, Conv1_NF = 128, Conv2_NF = 256,Conv3_NF = 128, lstmDropP = 0.8, FC_DropP = 0.3,device="cuda"):
        super(FCN_model,self).__init__()
        
        self.N_time = N_time
        self.N_Features = N_Features
        self.NumClassesOut = NumClassesOut
        self.N_LSTM_Out = N_LSTM_Out
        self.N_LSTM_layers = N_LSTM_layers
        self.Conv1_NF = Conv1_NF
        self.Conv2_NF = Conv2_NF
        self.Conv3_NF = Conv3_NF
        self.lstm = nn.LSTM(self.N_Features,self.N_LSTM_Out,self.N_LSTM_layers)
        self.C1 = nn.Conv1d(self.N_Features,self.Conv1_NF,8)
        self.C2 = nn.Conv1d(self.Conv1_NF,self.Conv2_NF,5)
        self.C3 = nn.Conv1d(self.Conv2_NF,self.Conv3_NF,3)
        self.BN1 = nn.BatchNorm1d(self.Conv1_NF)
        self.BN2 = nn.BatchNorm1d(self.Conv2_NF)
        self.BN3 = nn.BatchNorm1d(self.Conv3_NF)
        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(lstmDropP)
        self.ConvDrop = nn.Dropout(FC_DropP)
        self.FC = nn.Linear(self.Conv3_NF + self.N_LSTM_Out,self.NumClassesOut)
        self.device=device
    
    def init_hidden(self):
        
        h0 = torch.zeros(self.N_LSTM_layers, self.N_time, self.N_LSTM_Out).to(self.device)
        c0 = torch.zeros(self.N_LSTM_layers, self.N_time, self.N_LSTM_Out).to(self.device)
        return h0,c0
    
    def forward(self,x):
        
        # input x should be in size [B,T,F] , where B = Batch size
        #                                         T = Time sampels
        #                                         F = features
        
        h0,c0 = self.init_hidden()
        x1, (ht,ct) = self.lstm(x, (h0, c0))
        x1 = x1[:,-1,:]
        
        x2 = x.transpose(2,1)
        x2 = self.ConvDrop(self.relu(self.BN1(self.C1(x2))))
        x2 = self.ConvDrop(self.relu(self.BN2(self.C2(x2))))
        x2 = self.ConvDrop(self.relu(self.BN3(self.C3(x2))))
        x2 = torch.mean(x2,2)
        
        x_all = torch.cat((x1,x2),dim=1)
        x_out = self.FC(x_all)
        return x_out

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
            data_aimtrack.drop(data_aimtrack[data_aimtrack["exp_num"]==2].index,inplace=True)
            data_aimtrack.drop(["exp_num"],axis=1,inplace=True)
            """
            extract aimtrack data
            """
            if not (len(data_score)==len(data_eeg["epoch"].value_counts()) and len(data_score)==len(data_state[data_state["markerText"]=="ShotOps"]) and len(data_score)==len(data_aimtrack["shoot_num"].value_counts())): continue
            print("yes")
            for j in range(len(data_score)):
                if data_score[j]>9.5:
                    curaimtrackx=data_aimtrack[data_aimtrack["shoot_num"]==j+1]["x"][-40:].tolist()
                    curaimtracky=data_aimtrack[data_aimtrack["shoot_num"]==j+1]["y"][-40:].tolist()
                    if len(curaimtrackx)!=40 or len(curaimtracky)!=40: continue
                    datax.append([curaimtrackx,curaimtracky])
                    datay.append(0)
                    print(data_score[j])
                    high+=1

                if data_score[j]<7.5:
                    curaimtrackx=data_aimtrack[data_aimtrack["shoot_num"]==j+1]["x"][-40:].tolist()
                    curaimtracky=data_aimtrack[data_aimtrack["shoot_num"]==j+1]["y"][-40:].tolist()
                    if len(curaimtrackx)!=40 or len(curaimtracky)!=40: continue
                    datax.append([curaimtrackx,curaimtracky])
                    datay.append(1)
                    low+=1

            print(len(datay),len(datax))
            #if len(datay)!=len(datax):
            #    raise Exception("length not matched")
        except Exception as e:
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
    learning_rate=0.0001
    batch_size=2
    channel_size=1
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model=FCN_model(2,40,2).to(device)
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
            data=data.reshape(-1,40,2).to(device)
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
            test_x=test_x.reshape(-1,40,2).type(torch.FloatTensor).to(device)
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
    print(x[0],y[0])
    for i in range(len(dataxy[0])):
        print(len(dataxy[0][i]))
#show_input_data()
save_input_data()
#get_data()
#train_model()
#show_outcome(actual,predict)

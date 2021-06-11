from torch import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
import traceback
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import math
from glob import glob
import os


channel_names=None


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
        # input x should be in size [B,T,F] , where B = Batch size, T = Time sampels, F = features
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


class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.conv3=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1)
        self.mpool=nn.MaxPool2d(kernel_size=3)
        self.conv4=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)

        self.linear1=nn.Linear(384,100)
        self.linear2=nn.Linear(100,2)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=self.mpool(x)
        x=F.relu(self.conv4(x))
        x=x.view(-1,384)
        x=F.relu(self.linear1(x))
        x=F.relu(self.linear2(x))
        x = torch.sigmoid(x)
        return x

class ConvNet1(nn.Module):
    def __init__(self):
        super(ConvNet1,self).__init__()
        self.conv1=nn.Conv1d(in_channels=1,out_channels=5,kernel_size=1)
        self.conv2=nn.Conv1d(in_channels=5,out_channels=10,kernel_size=1)
        self.conv3=nn.Conv1d(in_channels=10,out_channels=20,kernel_size=1)
        self.linear1=nn.Linear(620,2)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=x.view(-1,620)
        x=F.relu(self.linear1(x))
        x=torch.sigmoid(x)
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


def get_data():
    global channel_names
    datax,datay=[],[]
    path=r"C:\Users\zhou\Desktop\毕业设计\mechanical\最终论文\pic\csv_t\-2_0\ttest\*"

    files=glob(path)
    highpsd,lowpsd=[],[]
    for file in tqdm(files):
        curpath=os.path.join(file,"*3.csv")
        bandfiles=glob(curpath)

        psd=[]
        for bandf in bandfiles:
            data=pd.read_csv(bandf,index_col=0)
            if not channel_names:
                channel_names=data["names"].tolist()
            psd.extend(data["psd"].tolist())
        datax.append(psd)
        if file[-4:]=="high":
            datay.append(1)
        if file[-3:]=="low":
            datay.append(0)
    
    datax_np=np.array(datax)
    datay_np=np.array(datay)
    print(datax_np.shape)
    print(datay_np.shape)
    return datax,datay

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
    learning_rate=0.0000001
    batch_size=5
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    #model=FCN_model(2,40,2).to(device)
    #model=ConvNet2().to(device)
    model=ConvNet1().to(device)
    dataxy=[]
    with open("D:/code/code/eegemotion/git/model/psd_new/psd_new.txt","rb") as f:
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
            data=data.reshape(-1,1,31).to(device)
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
            test_x=test_x.reshape(-1,1,31).type(torch.FloatTensor).to(device)
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
    
    torch.save(model.state_dict(),"D:/code/code/eegemotion/git/model/psd_new/model.pt")


def check_data():
    dataxy=[]
    with open("D:/code/code/eegemotion/git/model/psd_new/psd_new.txt","rb") as f:
        dataxy = pickle.load(f)
    x=np.array(dataxy[0])
    y=np.array(dataxy[1])

    train_x_origin,test_x_origin,train_y_origin,test_y_origin=train_test_split(x,y)
    print(x.shape)
    print(y.shape)
    print(x[0])
    print(y[0])


def save_input_data():
    dataxy=get_data()
    with open("D:/code/code/eegemotion/git/model/psd_new/psd_new.txt","wb") as f:
        pickle.dump(dataxy,f)

def show_input_data():
    dataxy=[]
    with open("D:/code/code/eegemotion/git/model/psd_new/psd_new.txt","rb") as f:
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
#save_input_data()
#get_data()
train_model()
#check_data()
#show_outcome(actual,predict)

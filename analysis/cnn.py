from eegprocess import get_epoch_eeg
from score import get_score
from state import get_state


from torch import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import time
import traceback


def get_data():
    for i in range(1,60):
        try:
            data_score=get_score(i)[0]
            data_state=get_state(i,5)
            data_eeg=get_epoch_eeg(i).drop(["condition"],axis=1)
            print(data_score)
            print(data_state)
            print(data_eeg)
            break
        except Exception as e:
            traceback.print_exc()
            break
            pass

def train_test_model():
    pass

def plot_loss():
    pass

get_data()
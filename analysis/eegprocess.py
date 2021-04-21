import mne
from glob import glob
import os
import configparser

def init():
    cf=configparser.ConfigParser()
    cf.read("path.conf")
    path_names=cf.options("path")
    for i in path_names:
        print(cf.get("path",i))

def preprocess():
    pass

    
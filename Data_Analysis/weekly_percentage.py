MODEL_PATH = "/home/jerometh/Models_on_datasets/Models/Anger_Go_Revised.pth"
DATA_DIR = "/home/jerometh/Tweet_Analysis/twitter_data/from_2018_cleaned"
IMAGE_DIR = "/home/jerometh/Tweet_Analysis/Anger_2019_Images_cleaned"
RESULTS_DIR = "/home/jerometh/Tweet_Analysis/Anger_2019_Results_cleaned"
DATA_FORM = "timeline3.json"
EMOTION = 'Anger'
MAX_LEN = 100
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 16
class_names = ['not_' + EMOTION, EMOTION]



import transformers
from transformers import BertModel, BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import pickle 
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import wandb
import json 
import csv
import time
import os
import Model as md
import times as tm
import os.path

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("-" * 50)
print(f'using {device}')

model = md.SentimentClassifier(len(class_names))
md.load_checkpoint(model, torch.load(MODEL_PATH)) # if model is saved as .pth file
# model.load_state_dict(torch.load(MODEL_PATH)) # if model is saved as .pkl file

model = model.to(device)

 
months = ["NULL",
         "January", 
         "February", 
         "March", 
         "April", 
         "May", 
         "June", 
         "July", 
         "August", 
         "September", 
          "October",
         "November", 
         "December"]

def out_of_range(date, data):
    return date < tm.get_date(data.iloc[0]['start_date']) or date > tm.get_date(data.iloc[-1]['end_date'])

def get_week(date, data):
    l = 0
    r = len(data) -1
    while l < r:
        mm = (l + r)//2
        now = data.loc[mm] # now is a dictionary
        st = tm.get_date(now['start_date'])
        ed = tm.get_date(now['end_date'])
        if ed < date:
            l = mm + 1
        else:
            r = mm
        
    return l


for filename in os.listdir(DATA_DIR):
    print(filename)
    
    if filename[0] != 't': continue
    print(filename[10: len(filename) - 4])
    STATE = filename[10: len(filename) - 4]
    if os.path.exists(f'{RESULTS_DIR}/{STATE}.json'): continue
    df = pd.read_csv(f'{DATA_DIR}/{filename}', encoding = "UTF8")
    data = pd.read_json(DATA_FORM)
    print("-" * 100)
    for i in range(len(df)):
        cur = df.loc[i]
        date = tm.get_date(cur['date'])
        
        if out_of_range(date, data) : continue
        pred = md.predict(model, cur['content'])

       
        week = get_week(date, data)
        data.at[week, 'count'] += pred
        data.at[week, 'total'] += 1
        if i % 5000 == 0: print(f'now at {i}')

    ma = 0
    for i in range(len(data)):
    
        if data['total'][i] == 0:
            continue
        else:
            data.at[i, 'percentage'] = data['count'][i]*100.00/data['total'][i]
            
        ma = max(ma, data['percentage'][i])
    
    with open(f'{RESULTS_DIR}/{STATE}.json', 'w') as outfile:
        json.dump(data.to_dict('records'), outfile)
    
   
        
    plt.plot(data['index'], data['percentage'])
    plt.title(f'{EMOTION} Tweet Percentage in {STATE} in Weeks from 2019/1/1')
    plt.ylabel(f'{EMOTION} Tweet Percentage')
    plt.xlabel('Weeks from 2019/1/1')
    plt.axis([0, 91, 0, ma * 1.2])
    plt.savefig(f'{IMAGE_DIR}/{STATE}_trend.jpg')
    plt.clf()

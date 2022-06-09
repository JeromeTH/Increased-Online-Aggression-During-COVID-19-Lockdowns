MODEL_PATH = "/home/jerometh/Models_on_datasets/Models/Offensive_AHSD.pth"


EMOTION = 'Offensive'
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
#setting device
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print("-" * 50)
print(f'using {device}')



tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)



class SentimentClassifier(nn.Module):
    
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p = 0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(
            input_ids = input_ids, 
            attention_mask = attention_mask
        ).pooler_output
        output = self.drop(pooled_output)
        return self.out(output)


def load_checkpoint(model, checkpoint):
    # checkpoint is a dictionary
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    print("=> Loading complete")

def predict(model, sen):
    encoded_sentence = tokenizer.encode_plus(
        sen,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoded_sentence['input_ids'].to(device)
    attention_mask = encoded_sentence['attention_mask'].to(device)
    output = model(input_ids, attention_mask)
    
    #print(output)
    _, prediction = torch.max(output, dim=1)

    #print(f'Review text: {sen}')
   # print(f'Sentiment  : {class_names[prediction]}')
    return prediction


def run_model(model, raw_tweets):
    offensive_count  = 0
    non_offensive_count = 0
    for tweet in raw_tweets:
        if predict(model, tweet) == 1: offensive_count += 1
        else: non_offensive_count += 1
    return offensive_count, non_offensive_count
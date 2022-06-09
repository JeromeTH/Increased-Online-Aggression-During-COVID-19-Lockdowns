MODEL_PATH = "Offensive_AHSD.pth"
DATA_PATH = "/home/jerometh/Models_on_datasets/Data/Offensive_ready.csv"

EMOTION = 'Offensive'
MAX_LEN = 100
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 16
class_names = ['not_' + EMOTION, 'EMOTION']


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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("-" * 50)
print(f'using {device}')

df = pd.read_csv(DATA_PATH, encoding='utf8')


tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


class CancerEmoDataset(Dataset):
    def __init__(self, sentences, targets, tokenizer, max_len):
        self.sentences = sentences
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'sentence': sentence,
            'input_ids': encoding['input_ids'].flatten(), ## look up this function,
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype = torch.long)
        }


    
    


    
    
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = CancerEmoDataset(
        sentences = df.Sentence.to_numpy(),
        targets = df[EMOTION].to_numpy(),
        tokenizer = tokenizer,
        max_len = max_len
    )
    
    return DataLoader(
        ds,
        batch_size = batch_size,
        num_workers = 0
    )

df_test = df[df['Split'] == 2]
data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)



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


model = SentimentClassifier(len(class_names))
model = model.to(device)


def load_checkpoint(checkpoint):
    # checkpoint is a dictionary
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    print("=> Loading complete")


load_checkpoint(torch.load(MODEL_PATH))



def get_predictions(model, data_loader):
    model = model.eval()
    sentences = []
    predictions = []
    prediction_probs = []
    real_values = []
    output = {}
    with torch.no_grad():
        for d in data_loader:
            sentence = d["sentence"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            sentences.extend(sentence)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(targets)
    
    #for i in range(10):
        #print(sentences[i])
    
        
        
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return sentences, predictions, prediction_probs, real_values





y_sentences, y_pred, y_pred_probs, y_test = get_predictions(
    model,
    data_loader
)

print("y_pred")
total = 0
correct = 0
with open("/home/jerometh/Models_on_datasets/output_files/AHSD_on_AHSD.csv", "w", encoding = 'utf8') as f:
    
    header = ["sentences", "Prediction", "Real_Value"]
    writer = csv.writer(f)
    writer.writerow(header)
    for i in range(len(y_sentences)):
        data = [y_sentences[i], y_pred[i].item(), y_test[i].item()]
        total += 1
        if y_pred[i].item() == y_test[i].item():
            correct += 1
        writer.writerow(data)
        
print("-" * 50)
print (f'Accuracy = {correct*100/total} %')
#with open("/home/jerometh/Sentiment_Classifier/output_files/output.txt", "w") as file:
        #json.dump(output, file)
        
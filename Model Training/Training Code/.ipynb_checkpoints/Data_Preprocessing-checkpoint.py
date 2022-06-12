DATA_PATH = "./Data/Hate.csv"
EMOTION = 'Hate'
MAX_LEN = 128
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
from sklearn.utils import shuffle 
import torch.nn.functional as F

#setting device 
#%matplotlib inline
#%config InlineBackend.figure_format='retina'

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
    
df_train = df[df['Split'] == 0]
df_val = df[df['Split'] == 1]
df_test = df[df['Split'] == 2]

df_train = shuffle(df_train)
df_val = shuffle(df_val)
df_test = shuffle(df_test)



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


def save_data_loader(filename, data_loader):
    with open(filename, 'wb') as file:
        pickle.dump(data_loader, file)

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

save_data_loader("./DataLoaders/train_data_loader.pkl", train_data_loader)
print("train loader saving complete")
save_data_loader("./DataLoaders/val_data_loader.pkl", val_data_loader)
save_data_loader("./DataLoaders/test_data_loader.pkl", test_data_loader)


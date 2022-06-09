
EMOTION = 'Hate'
MAX_LEN = 100
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 16
class_names = ['not_' + EMOTION, 'EMOTION']
training_config = {
    'fc_layer_size': 128,
    'dropout': 0.3,
    'epochs' : 10,
    'learning_rate': 2e-5,
    'pretrained_model' : PRE_TRAINED_MODEL_NAME
}
load_model = False
load_from = "Offensive_AHSD_OLID.pth"
load_path = "jerometh/Project_First_Round/2l8un3ip"
MODEL_NAME = "Hate_AHSD_LSCS"
save_to = MODEL_NAME + ".pth"
PROJECT_NAME = 'Hate_Models'


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



#loading data_loader
with open("./DataLoaders/val_data_loader.pkl", "rb") as file:
    val_data_loader = pickle.load(file)

with open("./DataLoaders/train_data_loader.pkl", "rb") as file:
    train_data_loader = pickle.load(file)

print("-" * 100)
print("Data Loader loaded")

'''for d in train_data_loader:
    data = d
    break'''
data = next(iter(train_data_loader))
data.keys()
print(data)

loss_fn = nn.CrossEntropyLoss().to(device)


#Creating model class
class SentimentClassifier(nn.Module):
    
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p = 0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)  # turn it into value using a linear algorithm 
        
    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(
            input_ids = input_ids, 
            attention_mask = attention_mask
        ).pooler_output
        output = self.drop(pooled_output)
        return self.out(output) # 


model = SentimentClassifier(len(class_names))
model = model.to(device)


#setting optimizer and scheduler 
input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

EPOCHS = training_config['epochs']
optimizer = AdamW(model.parameters(), lr=training_config['learning_rate'], correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

print("-" * 50)
print("optimizer and scheduler ready")

#training epoch and evaluating epoch
def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
    model = model.train()
    
    losses = []
    correct_predictions = 0
  
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)   # activation (max pooling )
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    print(f"[Training] Correct predictions: {correct_predictions.double()}, Total examples: {n_examples}")
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    
    print(f"[Validation] Correct predictions: {correct_predictions.double()}, Total examples: {n_examples}")
    return correct_predictions.double() / n_examples, np.mean(losses)



#saving and loading checkpoint

def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
    
def load_checkpoint(checkpoint):
    # checkpoint is a dictionary
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> Loading complete")



num_training_examples = len(data['sentence']) * len(train_data_loader)
num_val_examples = BATCH_SIZE * len(val_data_loader)
def train_and_save(config = None):
    with wandb.init(project = PROJECT_NAME, name = MODEL_NAME, config = config) as run:
        print("-" * 50)
        print(f'project activated')
        config = training_config
        total_steps = len(train_data_loader) * EPOCHS
        acc = 0
        if load_model:
            #best_model = wandb.restore(load_from, load_path)
            #load_checkpoint(torch.load(best_model))
            load_checkpoint(torch.load(load_from))

        for epoch in range(EPOCHS):
            print(f'Epoch {epoch + 1}/{EPOCHS}')
            print('-' * 10)
            if((epoch+1) %2== 0):
                checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                save_checkpoint(checkpoint, save_to)

            train_acc, train_loss = train_epoch(
                model,
                train_data_loader,    
                loss_fn, 
                optimizer, 
                device, 
                scheduler, 
                num_training_examples
            )
            #wandb.log({"loss": train_loss, "epoch": epoch})
            
            wandb.log({"Accuracy": train_acc, "Loss": train_loss, "epoch": epoch + 1})
            #f train_acc > acc:
                #cc = train_acc
                #odel.save(os.path.join(wandb.run.dir, "model.h5"))
            print(f'[Training] Loss: {train_loss} Accuracy: {train_acc}')

            val_acc, val_loss = eval_model(
                model,
                val_data_loader,
                loss_fn, 
                device, 
                num_val_examples
            )

            print(f'[Validation] Loss: {val_loss} Accuracy: {val_acc}')
            
            
        #logging artifact to wandb
        model_artifact = wandb.Artifact(
            "trained_model", type = "model", 
            description = f'Bert model trained with Sadness data with accuracy of {val_acc}', 
            metadata = dict(config))
        
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint, save_to)
        model_artifact.add_file(save_to)
        wandb.save(save_to)

        run.log_artifact(model_artifact)
        
        
        #what will happen to the chart if I run this tun two times? 
        

train_and_save()


        
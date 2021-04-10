import csv
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from konlpy.tag import Komoran
import random
import itertools
import re

stopwords= []
with open('./stopwords.csv', 'r', encoding = 'utf-8-sig') as File:
    csv_reader = csv.reader(File, delimiter = ',')
    for row in csv_reader:
        stopwords.append(row)
        

stopwords = list(itertools.chain(*stopwords))

def clean_str(text):
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+' # URL제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '<[^>]*>'         # HTML 태그 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '[^\w\s]'         # 특수기호제거
    text = re.sub(pattern=pattern, repl='', string=text)
    return text   

from torchtext import data

tokenizer = Komoran()
TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=tokenizer.morphs,
                  lower=True,
                  batch_first=True,
                  fix_length=60,
                 stop_words=stopwords
                 )

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True,
                  dtype = torch.float)


from torchtext.data import TabularDataset

train_data, test_data = TabularDataset.splits(path = '.', train = '욕설_train.csv', test = '욕설_test.csv', format = 'csv', fields = [('text', TEXT), ('label', LABEL)], skip_header = True)

len(train_data)

len(test_data)



USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)

# 하이퍼파라미터
batch_size = 100 
lr = 0.0009

TEXT.build_vocab(train_data, min_freq = 10, max_size = 20000)

len(TEXT.vocab)

train_data, val_data = train_data.split(split_ratio = 0.8)

train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data), batch_size=batch_size,
        shuffle=True, repeat=False, device = DEVICE, sort = False)

print('훈련 데이터의 미니 배치의 개수 : {}'.format(len(train_iter)))
print('테스트 데이터의 미니 배치의 개수 : {}'.format(len(test_iter)))
print('검증 데이터의 미니 배치의 개수 : {}'.format(len(valid_iter)))

class LSTM(nn.Module):
    def __init__(self, **model_config):
        super(LSTM, self).__init__()
        
        if model_config['emb_type'] == 'glove' or 'fasttext':
            self.emb = nn.Embedding(model_config['vocab_size'],
                                   model_config['emb_dim'],
                                   _weight = TEXT.vocab.vectors)
        else:
            self.emb = nn.Embedding(model_config['vocab_size'],
                                   model_config['emb_dim'])
            
        self.bidirectional = model_config['bidirectional']
        self.num_direction = 2 if model_config['bidirectional'] else 1
        self.model_type = model_config['model_type']
        
        self.LSTM = nn.LSTM(input_size = model_config['emb_dim'],
                           hidden_size = model_config['hidden_dim'],
                           dropout = model_config['dropout'],
                           bidirectional = model_config['bidirectional'],
                           batch_first = model_config['batch_first'])
        
        self.fc = nn.Linear(model_config['hidden_dim'] * self.num_direction,
                           model_config['output_dim'])
        
        self.drop = nn.Dropout(model_config['dropout'])
        
    def forward(self, x):
        emb = self.emb(x)
        output, (hidden, cell) = self.LSTM(emb)
        last_output = output[:,-1,:]
        
        return self.fc(self.drop(last_output))
    

model_config = dict(batch_first = True,
                        model_type = 'LSTM',
                        bidirectional = True,
                        hidden_dim = 256,
                        output_dim = 1,
                        dropout = 0.5,
                   emb_type = '',
                   vocab_size = len(TEXT.vocab),
                   emb_dim = 300,
                   batch_size = batch_size
                   )

model = LSTM(**model_config).to(DEVICE)

loss_fn = nn.BCEWithLogitsLoss().to(DEVICE)

optimizer = torch.optim.RMSprop(model.parameters(), lr)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum()/len(correct)
    return acc

def train(model, iterator, optimizer, loss_fn, idx_Epoch, **model_params):
    
    Epoch_loss = 0
    Epoch_acc = 0
    model.train()
    batch_size = model_params['batch_size']
    
    for idx, batch in enumerate(iterator):
        
        optimizer.zero_grad()
        
        predictions = model(next(iter(batch))).squeeze()
        loss = loss_fn(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        
        sys.stdout.write(
        "\r" + f"[Train] Epoch: {idx_Epoch + 1:^3}"\
            f"[{(idx + 1) * batch_size} / {len(iterator) * batch_size}({100. *(idx + 1) / len(iterator) :.4}%)]"\
            f"    Loss: {loss.item()}"\
            f"    Acc: {acc.item()}"\
        )
        
        # Backward
        loss.backward()
        optimizer.step()
        
        Epoch_loss += loss.item()
        Epoch_acc += acc.item()
        
    return Epoch_loss/len(iterator), Epoch_acc/len(iterator)

def evaluate(model, iterator, loss_fn):
    Epoch_loss = 0
    Epoch_acc = 0
    
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = loss_fn(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            
            Epoch_loss += loss.item()
            Epoch_acc += acc.item()
    
    return Epoch_loss/len(iterator), Epoch_acc/len(iterator)

N_EPOCH = 6
best_valid_loss = float('inf')
model_name = f"{'bi' if model_config['bidirectional'] else ''}{model_config['model_type']}_{model_config['emb_type']}"

print('-'*20)
print(f'Model name: {model_name}')
print('-'*20)

for Epoch in range(N_EPOCH):
    train_loss, train_acc = train(model, train_iter, optimizer, loss_fn, Epoch, **model_config)
    valid_loss, valid_acc = evaluate(model, valid_iter, loss_fn)
    print('')
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f'./{model_name}.pt')
        print(f'\t Saved at {Epoch}-Epoch')
    
    print(f'\t Epoch: {Epoch + 1} | Train Loss: {train_loss:.4} | Train Acc: {train_acc:.4}')
    print(f'\t Epoch: {Epoch + 1} | Valid Loss: {valid_loss:.4} | Valid Acc: {valid_acc:.4}')

test_loss, test_acc = evaluate(model, test_iter, loss_fn)
print('')
print(f'Test Loss: {test_loss:.4} | Test Acc: {test_acc:.4}')

def predict_sentiment(model, sentence, min_len = 0):
    model.eval().to(DEVICE)
    tokenized = [tok for tok in tokenizer.morphs(sentence)]
    print(tokenized)
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(DEVICE)
    tensor = tensor.unsqueeze(0)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

predict_sentiment(model, 'ㄱ ㅐ ㅅ ㅐ ㄲ ㅣ')
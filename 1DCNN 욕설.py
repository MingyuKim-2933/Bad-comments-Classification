import csv
import pandas as pd
import numpy as np
import torch
from konlpy.tag import Komoran
import torch.nn as nn
import torch.nn.functional as F
import itertools
from collections import defaultdict


stopwords= []
with open('./stopwords.csv', 'r', encoding = 'utf-8-sig') as File:
    csv_reader = csv.reader(File, delimiter = ',')
    for row in csv_reader:
        stopwords.append(row)
        

stopwords = list(itertools.chain(*stopwords))


csv.field_size_limit(100000000)
from torchtext import data

tokenizer = Komoran()
#데이터 필드 정의
TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=tokenizer.morphs,
                  lower=True,
                  batch_first=True,
                  fix_length=60,
                 stop_words = stopwords)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True,
                   dtype=torch.float)


from torchtext.data import TabularDataset


train_data, test_data = TabularDataset.splits(path='.', train='욕설_train.csv', test='욕설_test.csv', format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

print(vars(train_data[0]),vars(train_data[1])) #0번째 1번째 데이터 출력
print(train_data.shape)

TEXT.build_vocab(train_data, min_freq=5, max_size=20000) #최대 2만개, 5번 이상 나오는 단어로 단어사전
#unk=0, pad=1
vocab_size = len(TEXT.vocab)
print(vocab_size)
train_data, val_data = train_data.split(split_ratio=0.8) #train_data와 val_data에 train_data를 8:2로 나눔

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train_data, val_data, test_data), batch_size=100,
    shuffle=True, repeat=False, sort=False, device = device) #하나의 이터레이터에 batch_size만큼의 묶음으로 저장


######################CNN 모델
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.permute(0, 2, 1)

        # embedded = [batch size, emb dim, sent len]

        conved = [F.selu(conv(embedded)) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        cnn1 =  torch.tanh(cat)

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cnn1)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def accuracy(probs, target):
  predictions = probs.argmax(dim=1)
  corrects = (predictions == target)
  accuracy = corrects.sum().float() / float(target.size(0))
  return accuracy


import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

import torch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

import torch
import os
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
N_FILTERS = 50
FILTER_SIZES = [2,3,4,5,6]
#FILTER_SIZES = [2,3,4]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

###
# EMBEDDING_DIM = 100
# N_FILTERS = 100
# FILTER_SIZES = [3,4,5]
# OUTPUT_DIM = 1

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX).to(device)


model.convs
model

if __name__ == "__main__":
    
    print(device)
    model_type = "CNN"  # or: "token"

    data_type = "morph"  # or: "token"

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters()) #경사하강법 Adam
    N_EPOCHS = 5

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss, train_acc = train(model, train_iter, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, val_iter, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut4-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

model.load_state_dict(torch.load('tut4-model.pt'))

test_loss, test_acc = evaluate(model, test_iter, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

from torchtext import data

tokenizer = Komoran()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
def predict_sentiment(model, sentence, min_len = 60):
    model.eval().to(device)
    tokenized = [tok for tok in tokenizer.morphs(sentence)]
    print(tokenized)
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

predict_sentiment(model, "병신")
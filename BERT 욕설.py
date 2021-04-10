import torch
import torch.nn as nn
import sys
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, BertConfig
from torchtext import data
from torch.utils.data import Dataset, DataLoader
import re

device = torch.device('cuda')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

bert = BertModel.from_pretrained('bert-base-multilingual-cased')

max_input_length = tokenizer.max_model_input_sizes['bert-base-multilingual-cased']

def new_tokenizer(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:64]
    return tokens

def clean_str(text):
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+' # URL제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '<[^>]*>'         # HTML 태그 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '[^\w\s]'         # 특수기호제거
    text = re.sub(pattern=pattern, repl='', string=text)
    return text   

def PreProc(list_sentence):
    return [tokenizer.convert_tokens_to_ids(clean_str(x)) for x in list_sentence]

TEXT = data.Field(use_vocab=False,
                  tokenize=new_tokenizer,
                  lower=True,
                  batch_first=True,
                  preprocessing = PreProc,
                 init_token = tokenizer.cls_token_id,
                 eos_token = tokenizer.sep_token_id,
                 pad_token = tokenizer.pad_token_id,
                 unk_token = tokenizer.pad_token_id,
                 sequential = True)

LABEL = data.LabelField(dtype = torch.float)


from torchtext.data import TabularDataset



model_config = dict()
model_config['batch_first'] = True
model_config['model_type'] = 'BERT'
model_config['bidirectional'] = True
model_config['hidden_dim'] = 256
model_config['output_dim'] = 1
model_config['dropout'] = 0.5
model_config['emb_type'] = ''
model_config['batch_size'] = 40
model_config['lr'] = 3e-5


train_data, test_data = TabularDataset.splits(path = '.', train = '욕설new_train.csv', test = '욕설_test.csv', format = 'csv', fields = [('text', TEXT), ('label', LABEL)], skip_header = True)

train_data, val_data = train_data.split(split_ratio = 0.8)

LABEL.build_vocab(train_data)

train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data), batch_size=model_config['batch_size'],
        device = device, sort = False, repeat = False, shuffle = True)

len(test_iter)

model_config['emb_dim'] = bert.config.to_dict()['hidden_size']

class SentenceClassification(nn.Module):
    def __init__(self, **model_config):
        super(SentenceClassification, self).__init__()
        self.bert = bert
        self.fc = nn.Linear(model_config['emb_dim'],
                           model_config['output_dim'])
    
    def forward(self, x):
        pooled_cls_output = self.bert(x)[1]
        return self.fc(pooled_cls_output)

model = SentenceClassification(**model_config).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = model_config['lr'])

loss_fn = nn.BCEWithLogitsLoss().to(device)

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
        "\r" + f"[Train] Epoch: {idx_Epoch+1:^3}"\
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

N_EPOCH = 5

best_valid_loss = float('inf')
model_name = 'BERT'
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
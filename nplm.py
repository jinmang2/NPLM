import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

dtype = torch.FloatTensor

def load_datasets(path=''):
    with open(path + 'datasets.txt', 'r', encoding='utf-8') as f:
        datasets = f.readlines()
        datasets = ''.join(datasets).split('\n')
    return datasets

def add_unk_token(datasets, NGRAM_SIZE, unk_token="<UNK>"):
    for i, sen in enumerate(datasets):
        word_list = sen.split()
        if len(word_list) < NGRAM_SIZE:
            word_list.insert(0, unk_token)
            sen = ' '.join(word_list)
        datasets[i] = sen
        
def make_ngram_data(datasets, NGRAM_SIZE):
    train_data = []
    for sen in datasets:
        text = sen.strip().split(' ')
        for i in range(len(text) - (NGRAM_SIZE - 1)):
            train_data.append(' '.join(text[i:i+NGRAM_SIZE]))
    return train_data

def make_batch_generator(sentences, batch_size):
    input_batch = []
    target_batch = []
    batch_ix = 0
    for sentence in sentences[batch_ix:]:
        if batch_ix == batch_size:
            input_batch = Variable(torch.LongTensor(input_batch))
            target_batch = Variable(torch.LongTensor(target_batch))
            yield input_batch, target_batch
            input_batch = []
            target_batch = []
            batch_ix = 0
        words = sentence.split()
        input_ = [word_dict[n] for n in words[:-1]]
        target_ = word_dict[words[-1]]
        input_batch.append(input_)
        target_batch.append(target_)
        batch_ix += 1
        
class NPLM(nn.Module):
    
    def __init__(self,
                 VOCAB_SIZE,
                 EMBED_SIZE=30,
                 HIDDEN_SIZE=32,
                 NGRAM_SIZE=2):
        super(NPLM, self).__init__()
        self.C = nn.Embedding(VOCAB_SIZE, EMBED_SIZE).cuda()
        self.H = nn.Parameter(torch.randn((NGRAM_SIZE-1)*EMBED_SIZE, HIDDEN_SIZE).type(dtype)).cuda()
        self.W = nn.Parameter(torch.randn((NGRAM_SIZE-1)*EMBED_SIZE, VOCAB_SIZE).type(dtype)).cuda()
        self.d = nn.Parameter(torch.randn(HIDDEN_SIZE).type(dtype)).cuda()
        self.U = nn.Parameter(torch.randn(HIDDEN_SIZE, VOCAB_SIZE).type(dtype)).cuda()
        self.b = nn.Parameter(torch.randn(VOCAB_SIZE).type(dtype)).cuda()
        
    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, n_step * m)
        tanh = torch.tanh(self.d + torch.mm(X, self.H))
        output = self.b + torch.mm(X, self.W) + torch.mm(tanh, self.U)
        return output

def main():
    datasets = load_datasets()
    NUM_SENTENCES = len(datasets)
    word_list = " ".join(datasets).split()
    word_list = list(set(word_list))
    word_list.append("<UNK>")
    
    word_dict = {w:i for i, w in enumerate(word_list)}
    number_dict = {i:w for i, w in enumerate(word_list)}
    
    VOCAB_SIZE = len(word_dict)
    EMBED_SIZE = 30
    HIDDEN_SIZE = 32
    NGRAM_SIZE = 3
    BATCH_SIZE = 32
    
    add_unk_token(datasets, NGRAM_SIZE)
    train_data = make_ngram_data(datasets, NGRAM_SIZE)
    
    model = NPLM(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NGRAM_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(5000):
        train_generator = make_batch_generator(train_data, BATCH_SIZE)
        for input_batch, target_batch in train_generator:
            optimizer.zero_grad()
            output_batch = model(input_batch)
            loss = criterion(output_batch, target_batch)
            loss.backward()
            optimizer.step()

# Neural Probabilistic Language Model

```python
# Library 호출
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

dtype = torch.FloatTensor

# dataset read, wiki 한글 속담 데이터
with open('datasets.txt', 'r', encoding='utf-8') as f:
    datasets = f.readlines()
    datasets = ''.join(datasets).split('\n')

datasets[::40]
>>> ['발 없는 말이 천리 간다',
>>>  '모난 돌이 정 맞는다',
>>>  '고래 싸움에 새우 등 터진다',
>>>  '끝 부러진 송곳',
>>>  '들으면 병이요  안 들으면 약이다',
>>>  '봄눈 녹듯 한다',
>>>  '열 손가락을 깨물어서 안 아픈 손가락 없다',
>>>  '첫딸은 살림 밑천이다']

len(datasets)
>>> 315

# 단어 사전 구축
word_list = " ".join(datasets).split()
word_list = list(set(word_list))
unk_token = '<UNK>'
word_list.append(unk_token)

# 단어 to ix, ix to 단어 사전 구축
word_dict = {w:i for i, w in enumerate(word_list)}
number_dict = {i:w for i, w in enumerate(word_list)}

# Hyper-Parameter settings
VOCAB_SIZE = len(word_dict)
EMBED_SIZE = 30
HIDDEN_SIZE = 32
NGRAM_SIZE = 3
BATCH_SIZE = 32

print('VOCAB_SIZE  :', VOCAB_SIZE)
print('EMBED_SIZE  :', EMBED_SIZE)
print('HIDDEN_SIZE :', HIDDEN_SIZE)
print('NGRAM_SIZE  :', NGRAM_SIZE)
print('BATCH_SIZE  :', BATCH_SIZE)
>>> VOCAB_SIZE  : 1008
>>> EMBED_SIZE  : 30
>>> HIDDEN_SIZE : 32
>>> NGRAM_SIZE  : 3
>>> BATCH_SIZE  : 32

# 두 단어(ex> 인생은 쓰다)의 경우, <UNK> token을 넣어줌.
for i, sen in enumerate(datasets):
    word_list = sen.split()
    if len(word_list) < NGRAM_SIZE:
        word_list.insert(0, unk_token)
        sen = ' '.join(word_list)
    datasets[i] = sen

# ' '*2 -> ' '
datasets = [sen.replace('  ', ' ') for sen in datasets]

# N-Gram
train_data = []
for sen in datasets:
    text = sen.strip().split(' ')
    for i in range(len(text) - (NGRAM_SIZE - 1)):
        train_data.append(' '.join(text[i:i+NGRAM_SIZE]))

# input/target batch generator 생성
def make_batch_generator(sentences, batch_size):
    input_batch = []
    target_batch = []
    for ix, sentence in enumerate(sentences):
        if (ix + 1) % batch_size == 0:
            input_batch = Variable(torch.LongTensor(input_batch)).cuda()
            target_batch = Variable(torch.LongTensor(target_batch)).cuda()
            yield input_batch, target_batch
            input_batch = []
            target_batch = []
        words = sentence.split()
        input_ = [word_dict[n] for n in words[:-1]]
        target_ = word_dict[words[-1]]
        input_batch.append(input_)
        target_batch.append(target_)

# NPLM Class @그분
class NPLM(nn.Module):

    def __init__(self,
                 VOCAB_SIZE,
                 EMBED_SIZE=30,
                 HIDDEN_SIZE=32,
                 NGRAM_SIZE=2):
        super(NPLM, self).__init__()
        self.VOCAB_SIZE = VOCAB_SIZE
        self.EMBED_SIZE = EMBED_SIZE
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.NGRAM_SIZE = NGRAM_SIZE
        self.C = nn.Embedding(VOCAB_SIZE, EMBED_SIZE).cuda()
        self.H = nn.Parameter(
            torch.randn((NGRAM_SIZE-1)*EMBED_SIZE, HIDDEN_SIZE).type(dtype)).cuda()
        self.W = nn.Parameter(
            torch.randn((NGRAM_SIZE-1)*EMBED_SIZE, VOCAB_SIZE).type(dtype)).cuda()
        self.d = nn.Parameter(
            torch.randn(HIDDEN_SIZE).type(dtype)).cuda()
        self.U = nn.Parameter(
            torch.randn(HIDDEN_SIZE, VOCAB_SIZE).type(dtype)).cuda()
        self.b = nn.Parameter(
            torch.randn(VOCAB_SIZE).type(dtype)).cuda()

    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, (self.NGRAM_SIZE - 1) * self.EMBED_SIZE)
        tanh = torch.tanh(self.d + torch.mm(X, self.H))
        output = self.b + torch.mm(X, self.W) + torch.mm(tanh, self.U)
        return output

# model 호출
model = NPLM(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NGRAM_SIZE)

# parameters 메서드를 호출하면 C만 뜸
list(model.parameters()), len(list(model.parameters()))
>>> ([Parameter containing:
>>>   tensor([[ 1.1638, -0.0951, -0.0233,  ..., -1.9472,  0.5449, -0.9794],
>>>           [ 1.9428,  0.0899, -0.1759,  ..., -1.4997,  0.5290,  0.1428],
>>>           [-1.2909, -3.0661,  0.5343,  ...,  0.2385, -0.0742, -0.2328],
>>>           ...,
>>>           [-1.1278, -0.7776,  1.3159,  ..., -1.0650,  0.0154,  0.1327],
>>>           [-0.0613, -0.3044,  0.0067,  ...,  0.0443, -0.7996,  1.2539],
>>>           [-0.1671,  0.8982, -0.4620,  ...,  2.3737,  1.9044, -1.5356]],
>>>          device='cuda:0', requires_grad=True)], 1)

model.H
>>> tensor([[-1.6174e+00, -1.3252e+00, -1.2787e+00,  ...,  2.2481e-01,
>>>          -1.8602e+00,  1.3455e+00],
>>>         [-8.5359e-01, -5.9634e-01, -3.5102e-04,  ..., -2.2890e+00,
>>>          -1.1117e+00, -7.7680e-01],
>>>         [ 5.2114e-01,  2.2067e-01,  2.0605e-02,  ...,  2.3825e-01,
>>>           1.6374e-01, -7.6036e-01],
>>>         ...,
>>>         [-3.3564e-01,  5.6876e-01, -7.9408e-01,  ...,  1.4366e+00,
>>>          -7.4906e-01,  1.6060e+00],
>>>         [ 5.8770e-01,  8.0177e-01,  3.1561e-01,  ..., -2.6691e-01,
>>>           1.3397e-01,  3.2455e-01],
>>>         [ 5.4305e-01, -2.1393e-01,  4.7845e-01,  ..., -1.2968e+00,
>>>          -6.9006e-01,  7.3234e-01]], device='cuda:0', grad_fn=<CopyBackwards>)

model.W
>>> tensor([[-0.1996,  1.0247, -0.6503,  ...,  2.0768, -0.4768,  0.8306],
>>>         [-1.3389, -0.1676, -0.8898,  ...,  1.7507,  0.8217, -0.2155],
>>>         [-2.0034,  1.8582,  0.7308,  ...,  0.0999, -0.7425, -3.0941],
>>>         ...,
>>>         [ 0.4080, -1.8451, -1.8541,  ..., -1.6243, -0.8306,  1.3033],
>>>         [ 0.3135,  0.6711, -1.5834,  ...,  1.3785,  0.2687, -0.5657],
>>>         [ 0.9939,  1.9021, -0.2424,  ...,  0.6669, -0.7991, -0.2573]],
>>>        device='cuda:0', grad_fn=<CopyBackwards>)

model.U
>>> tensor([[-0.1952,  1.0994,  1.6964,  ..., -0.3940, -0.3129,  0.8586],
>>>         [-0.5009, -0.0287,  1.8413,  ..., -1.9629,  0.6003, -1.7462],
>>>         [-0.2664,  0.4922,  0.1442,  ..., -1.4465,  0.0407,  0.3524],
>>>         ...,
>>>         [-0.0158,  0.6753,  0.3638,  ..., -0.5692,  1.4864, -0.6355],
>>>         [-2.1896, -0.9145,  1.7770,  ...,  1.7305, -0.9666,  0.3723],
>>>         [ 1.3943,  1.2133,  0.3527,  ..., -1.8678, -0.1180, -0.3532]],
>>>        device='cuda:0', grad_fn=<CopyBackwards>)

# Cross-Entropy Loss, Adam 선언
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습
%%time
for epoch in range(1000):
    train_generator = make_batch_generator(train_data, BATCH_SIZE)
    for input_batch, target_batch in train_generator:
        optimizer.zero_grad()
        output_batch = model(input_batch)
        loss = criterion(output_batch, target_batch)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 50 == 0:
        print('Epoch:', '%4d' % (epoch + 1), 'cost = {:.6f}'.format(loss))
>>> Epoch:   50 cost = 18.702868
>>> Epoch:  100 cost = 11.424132
>>> Epoch:  150 cost = 5.745163
>>> Epoch:  200 cost = 2.551628
>>> Epoch:  250 cost = 1.297873
>>> Epoch:  300 cost = 0.784024
>>> Epoch:  350 cost = 0.309876
>>> Epoch:  400 cost = 0.078873
>>> Epoch:  450 cost = 0.057772
>>> Epoch:  500 cost = 0.050481
>>> Epoch:  550 cost = 0.048012
>>> Epoch:  600 cost = 0.045561
>>> Epoch:  650 cost = 0.044771
>>> Epoch:  700 cost = 0.044341
>>> Epoch:  750 cost = 0.044078
>>> Epoch:  800 cost = 0.043905
>>> Epoch:  850 cost = 0.043783
>>> Epoch:  900 cost = 0.043676
>>> Epoch:  950 cost = 0.043599
>>> Epoch: 1000 cost = 0.043560
>>> Wall time: 59.9 s

# C 행렬은 업데이트됨.
list(model.parameters()), len(list(model.parameters()))
>>> ([Parameter containing:
>>>   tensor([[ 1.5509, -0.5137, -1.0573,  ..., -1.4310,  1.5886, -0.5866],
>>>           [ 1.3439, -0.5930,  0.3124,  ..., -2.1737,  0.3886, -0.4301],
>>>           [-1.2909, -3.0661,  0.5343,  ...,  0.2385, -0.0742, -0.2328],
>>>           ...,
>>>           [-2.2934, -1.8352,  0.8021,  ..., -0.5497,  1.1274, -0.1156],
>>>           [ 1.5847, -0.1359,  1.6275,  ...,  0.4208, -1.1859,  1.4233],
>>>           [-0.1131,  0.3847, -0.6870,  ...,  0.7912,  0.7540,  0.2095]],
>>>          device='cuda:0', requires_grad=True)], 1)

# 그러나 아래 H, W, U 행렬은 초기값과 동일함.
model.H
>>> tensor([[-1.6174e+00, -1.3252e+00, -1.2787e+00,  ...,  2.2481e-01,
>>>          -1.8602e+00,  1.3455e+00],
>>>         [-8.5359e-01, -5.9634e-01, -3.5102e-04,  ..., -2.2890e+00,
>>>          -1.1117e+00, -7.7680e-01],
>>>         [ 5.2114e-01,  2.2067e-01,  2.0605e-02,  ...,  2.3825e-01,
>>>           1.6374e-01, -7.6036e-01],
>>>         ...,
>>>         [-3.3564e-01,  5.6876e-01, -7.9408e-01,  ...,  1.4366e+00,
>>>          -7.4906e-01,  1.6060e+00],
>>>         [ 5.8770e-01,  8.0177e-01,  3.1561e-01,  ..., -2.6691e-01,
>>>           1.3397e-01,  3.2455e-01],
>>>         [ 5.4305e-01, -2.1393e-01,  4.7845e-01,  ..., -1.2968e+00,
>>>          -6.9006e-01,  7.3234e-01]], device='cuda:0', grad_fn=<CopyBackwards>)

model.W
>>> tensor([[-0.1996,  1.0247, -0.6503,  ...,  2.0768, -0.4768,  0.8306],
>>>         [-1.3389, -0.1676, -0.8898,  ...,  1.7507,  0.8217, -0.2155],
>>>         [-2.0034,  1.8582,  0.7308,  ...,  0.0999, -0.7425, -3.0941],
>>>         ...,
>>>         [ 0.4080, -1.8451, -1.8541,  ..., -1.6243, -0.8306,  1.3033],
>>>         [ 0.3135,  0.6711, -1.5834,  ...,  1.3785,  0.2687, -0.5657],
>>>         [ 0.9939,  1.9021, -0.2424,  ...,  0.6669, -0.7991, -0.2573]],
>>>        device='cuda:0', grad_fn=<CopyBackwards>)

model.U
>>> tensor([[-0.1952,  1.0994,  1.6964,  ..., -0.3940, -0.3129,  0.8586],
>>>         [-0.5009, -0.0287,  1.8413,  ..., -1.9629,  0.6003, -1.7462],
>>>         [-0.2664,  0.4922,  0.1442,  ..., -1.4465,  0.0407,  0.3524],
>>>         ...,
>>>         [-0.0158,  0.6753,  0.3638,  ..., -0.5692,  1.4864, -0.6355],
>>>         [-2.1896, -0.9145,  1.7770,  ...,  1.7305, -0.9666,  0.3723],
>>>         [ 1.3943,  1.2133,  0.3527,  ..., -1.8678, -0.1180, -0.3532]],
>>>        device='cuda:0', grad_fn=<CopyBackwards>)
```

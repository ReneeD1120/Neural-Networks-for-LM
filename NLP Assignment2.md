# Reports of NLP Assignment  (22-23 Autumn)

This report is divided into 3 parts : NNLM,RNNLM, BERT. In this experiment, I mainly use cross entropy loss to compute perpexity for conveniece.

Cross entropy loss of language models of size T:
$$
J^{(t)}(\theta)=-\frac{1}{T}\sum^{T}_{t=1}\sum^{|V|}_{j=1}y_{t,j}\times log(\hat y_{t,j})
$$
Perplexity of language models of size T:
$$
Perplexity=2^J
$$


## 1.NNLM

### Code

Import package and preprocess data.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import re
import numpy as np
import math
with open('/content/drive/MyDrive/英文文档NLP/data.txt','r',encoding='utf-8') as file:
    content=file.readlines()
def preprocessing(sent):
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = "".join(pattern.findall(sent))
    #chars =[w for w in chars if len(w) > 0]
    return chars
  
```

```python
def dic(content):
  #sentences=preprocessing(content)
  
  word_list=preprocessing(content)
  #word_list=jieba.lcut(word_list)
  word_list = ' '.join(word_list).split()
  word_list = list(set(word_list))

  word2id = {w: i for i, w in enumerate(word_list)}
  id2word = {i: w for i, w in enumerate(word_list)}
  return word2id
```

Build a dictionary and make batches, and then build a new feedforward neural network and train it.

```python
with open('/content/drive/MyDrive/英文文档NLP/data.txt','r',encoding='utf-8') as file:
    content_total=file.read()
    word2id=dic(content_total)
    
def computeCE(sen,word2id):

    
    
  #sentences = ['我 喜欢 狗', '我 讨厌 牛奶', '我 做 自然语言处理', '我 看 电影']
  sentences = sen
  
  #word_list = ' '.join(sentences).split()
  #word_list = list(set(word_list))

  #word2id = {w: i for i, w in enumerate(word_list)}
  #id2word = {i: w for i, w in enumerate(word_list)}

  def make_batch(sentence):
      input_batch = []
      target_batch = []
      #word = sentence.split()
      
      for sen in sentence:
        #word=jieba.lcut(sen)
        word=' '.join(sen).split()
        for n in range(len(word)-5):
          inputs = [word2id[w] for w in word[n:n+5]]
          target = word2id[word[n+5]]
          input_batch.append(inputs)
          target_batch.append(target)
      #print(input_batch,target_batch)
      return input_batch, target_batch

  input_batch, target_batch = make_batch(sentences)
  input_batch = torch.LongTensor(input_batch)
  target_batch = torch.LongTensor(target_batch)
  input_batch, target_batch = make_batch(sentences)
  input_batch = torch.LongTensor(input_batch)
  target_batch = torch.LongTensor(target_batch)


  n_class = len(word2id) # 词表大小
  n_hidden = 10   # 隐藏层神经单元数目
  m = 5  # 词向量维度
  n_step = 5   # 代表n-gram模型 2就是2-gram 用前两个词预测最后一个词
  # 搭建模型
  class NNLM(nn.Module):
      def __init__(self):
          super(NNLM, self).__init__()
          self.embd = nn.Embedding(n_class, m)
          self.W = nn.Parameter(torch.randn(n_step*m, n_hidden))
          self.b = nn.Parameter(torch.randn(n_hidden))
          self.U = nn.Parameter(torch.randn(n_hidden, n_class))
          self.d = nn.Parameter(torch.randn(n_class))

      def forward(self, x):
          x = self.embd(x)
          x = x.view(-1, n_step*m)
          tanh = torch.tanh(torch.mm(x, self.W) + self.b)
          output = torch.mm(tanh, self.U) + self.d

          return output

  model = NNLM()  # 模型实例化
  criterion = nn.CrossEntropyLoss()  # 选择损失函数
  optimizer = optim.Adam(model.parameters(), lr=0.001)  # 选择优化器

  # 迭代训练
  for i in range(1000):
      optimizer.zero_grad()  # 梯度清零

      output = model(input_batch)
      
      loss = criterion(output, target_batch)
      loss.backward()
      optimizer.step()


  #predict = model(input_batch).data.max(1, keepdim=True)[1]
  #print([sen.split()[:-1] for sen in sentences], '-->', [id2word[n.item()] for n in predict.squeeze()])
      
  return loss,model
```

Cross Entropy loss is chosen for convenience to compute perplexity in the next step.

```python
ppltotal=0.
count=0
sen1=[]
for i in range(len(content)):
  sen=preprocessing(content[i])
  if len(sen)>5:
    sen1.append(sen)
train_data=sen1[0:20000]

CE,model=computeCE(train_data,word2id)
print(CE.item())

def make_batch(sentence):
    input_batch = []
    target_batch = []
      #word = sentence.split()
      
    for sen in sentence:
        #word=jieba.lcut(sen)
      word=' '.join(sen).split()
      for n in range(len(word)-5):
        inputs = [word2id[w] for w in word[n:n+5]]
        target = word2id[word[n+5]]
        input_batch.append(inputs)
        target_batch.append(target)
      #print(input_batch,target_batch)
    return input_batch, target_batch
    
testdata=sen1[20000:30000]
loss=[]
for test_data in testdata:
  input_batch, target_batch = make_batch(test_data)
  input_batch = torch.LongTensor(input_batch)
  target_batch = torch.LongTensor(target_batch)
  criterion = nn.CrossEntropyLoss()
  l=criterion(model(input_batch),target_batch)
  print(l)
  loss.append(l)
testdata=sen1[20000:50000]
loss=[]

input_batch, target_batch = make_batch(testdata)
input_batch = torch.LongTensor(input_batch)
target_batch = torch.LongTensor(target_batch)
criterion = nn.CrossEntropyLoss()
l=criterion(model(input_batch),target_batch)
print(l)
loss.append(l)
ppl=sum(np.exp(loss[i].item()) for i in range(len(loss)))
```

It's a 5-gram NN language model. 20000 lines were used in this experiment as train data, while 10000 lines as test data. Computing time for training models is long, so in the other two models I choose smaller train data size. 

### Results

#### Model Parameters

```python
OrderedDict([('W',
              tensor([[-1.5074, -0.1107, -1.0600, -0.5311,  0.2881,  1.1077, -1.8167, -1.2301,
                        1.0879,  1.9528],
                      [ 1.2777,  0.8877,  0.3649,  1.2157, -0.0506, -0.6590,  1.4483,  1.7335,
                       -1.0046, -1.0593],
                      [ 0.4919, -0.7020, -1.4984,  0.3140,  0.3932,  0.9912, -0.5077, -0.3617,
                        1.4427,  0.7543],
                      [-0.4075,  1.1610, -0.2447, -0.3880,  0.3247,  0.3473, -0.1689, -0.2811,
                       -0.1880,  0.2287],
                      [-0.3022, -0.5945, -1.1386, -0.1535, -0.5180,  0.4106,  0.9354, -0.6607,
                        0.9835,  0.4959],
                      [-0.9134, -0.7141, -0.7998, -0.0647,  1.5979,  1.3476, -0.6137, -0.5599,
                        1.2668,  0.5875],
                      [ 0.2867,  0.5220,  1.0815,  0.4005, -1.5239, -1.0870,  0.0503,  0.5875,
                       -0.5302,  0.1804],
                      [-0.6896,  0.3049, -0.8499, -0.3827,  0.8417, -0.0037, -0.8387, -0.4924,
                        0.8064,  1.2421],
                      [-0.7231, -0.0301, -0.3587, -0.0314,  1.3514,  0.0920, -0.4714,  0.5218,
                       -0.5635, -0.6338],
                      [ 0.2226, -0.2912, -0.9520,  0.0624,  0.3061, -0.2835, -0.0928,  0.3076,
                        0.5930,  0.4858],
                      [-0.8564, -1.3957, -1.6045, -0.7466,  0.6635,  0.8995,  0.1356, -1.0960,
                        1.1613,  0.6221],
                      [ 0.9946,  2.1084,  0.5871,  0.8834, -0.2923, -0.7162,  0.4841,  1.3782,
                       -0.4803, -1.2376],
                      [-1.1446, -0.1224,  0.1898,  0.4675,  1.3828,  2.0012, -0.1265, -0.0967,
                       -0.5935,  1.2407],
                      [-0.3912, -0.6368, -1.4146, -0.4162,  0.2690,  0.5500,  0.2077, -0.0275,
                        0.5684,  0.7097],
                      [-1.2429, -0.7751, -0.2467, -0.4935,  0.6433,  0.4487, -0.6276, -0.9012,
                        0.5229,  1.0513],
                      [-0.6990, -1.0805, -1.2093, -0.6207,  0.4005, -0.4090, -0.9921, -1.3327,
                        0.8834,  0.9500],
                      [-0.7811,  0.3793,  1.0357, -0.6166, -1.0657, -0.2630, -0.8220,  0.5192,
                       -0.7720,  0.1942],
                      [ 1.0447, -0.3593, -1.0619, -0.5720,  1.1598,  0.5232, -0.4697,  0.5842,
                       -0.2228, -0.0205],
                      [ 0.3622, -0.0445, -0.0322, -0.4033,  0.4370,  0.3493, -0.5258, -0.8660,
                        0.3260,  0.8029],
                      [ 0.5156, -0.3347, -0.4512, -0.3467,  0.9972,  0.0681, -0.4439,  0.2645,
                        0.4090,  0.6021],
                      [-1.7891, -0.0835,  1.6178, -0.1559,  0.5779,  1.4835, -0.7023, -0.9481,
                        0.7512,  1.9283],
                      [ 0.4981,  0.2393,  1.2816,  0.3840,  0.3023,  1.5246, -0.1865,  1.2010,
                       -0.4789,  2.4720],
                      [-1.6571, -0.7714,  1.2921, -0.5838,  0.0680,  1.2149, -0.8836, -0.5671,
                        0.1971,  1.8544],
                      [-1.4399, -0.9172,  0.3964, -0.6613,  0.0138,  0.7435, -0.6961, -1.0081,
                       -0.4789,  1.2665],
                      [-0.7374, -0.2789, -0.8703, -0.0516,  1.4131,  1.4380, -0.4787,  0.9032,
                        1.0719,  1.5468]])),
             ('b',
              tensor([ 0.4631,  1.7737,  0.6065,  1.2010, -1.4526, -0.6199,  2.2314,  0.7166,
                      -1.5390, -1.5363])),
             ('U',
              tensor([[-0.8125,  0.2811, -1.5372,  ...,  0.3958,  1.4722, -0.8124],
                      [-1.0802,  2.7714, -0.3473,  ..., -0.9472, -1.6997, -0.6455],
                      [ 0.4309,  1.3584, -0.7345,  ..., -0.4247,  0.1360, -0.3145],
                      ...,
                      [-0.1850, -0.7611,  0.0797,  ...,  1.2608, -1.1994,  0.7733],
                      [-1.8858, -0.6412,  0.5905,  ...,  0.0340,  0.7301, -0.6736],
                      [ 0.0190,  0.9032,  1.3505,  ..., -0.1567, -0.6500, -0.0311]])),
             ('d',
              tensor([ 1.0757, -0.1223,  0.5951,  ...,  1.0592,  0.0332, -1.3731])),
             ('embd.weight',
              tensor([[-0.0545, -0.4246, -0.2535,  0.6641, -0.1480],
                      [-2.0199, -0.2482, -0.1215, -0.1434, -1.1727],
                      [-0.2941, -0.3317, -0.7034,  1.5165, -1.1570],
                      ...,
                      [-0.1642, -0.1446, -0.8807, -0.6884, -0.3040],
                      [-0.4526,  0.4182,  0.1291,  0.1830, -1.9619],
                      [-0.4845, -1.6751,  0.9974,  1.0663,  1.6294]]))])
```

#### Perplexity

```
873.6391407766108
```

## 2. RNNLM

### Code

Import packages and preprocessing data.

```python
import torchtext
from torchtext.vocab import Vectors
import torch
import torch.nn as nn
import numpy as np
import random
import re
from torchtext.legacy.data import Field
def preprocessing(sent):
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = "".join(pattern.findall(sent))
    #chars =[w for w in chars if len(w) > 0]
    return chars

with open('/content/drive/MyDrive/英文文档NLP/data.txt','r',encoding='utf-8') as file:
    content=file.readlines()
sen1=[]
for i in range(len(content)):
  sen=preprocessing(content[i])
  if len(sen)>0:
    sen1.append(sen)
with open('train.txt','w',encoding='utf-8') as file:
  for s in sen1[0:1000]:
    file.writelines(s+'\n')
  file.close

with open('dev.txt','w',encoding='utf-8') as file:
  for s in sen1[1000:1500]:
    file.write(s+'\n')
    file.close
  
with open('test.txt','w',encoding='utf-8') as file:
  for s in sen1[1500:2000]:
    file.write(s+'\n')
    file.close
#np.savetxt('dev.txt',str(sen1[1000:1500]))
#np.savetxt('test.txt',sen1[1500:2000])

batch_size = 32
embedding_size = 50
max_vocab_size = 50000
hidden_size = 100
learn_rate = 0.001

TEXT = Field(lower=True)
# 专门用来处理语言模型数据集
train,val,test=torchtext.legacy.datasets.LanguageModelingDataset.splits(path=".",  #当前文件夹
                    train="train.txt", 
                    validation="dev.txt", 
                    test="test.txt",text_field=TEXT)
TEXT.build_vocab(train,max_size=max_vocab_size)          # 从train数据集中建立词表，按词频建立
# print(TEXT.vocab.itos[:100])
# print(TEXT.vocab.stoi['<unk>'])
 
train_iter,val_iter,test_iter = torchtext.legacy.data.BPTTIterator.splits((train,val,test),
                                batch_size=batch_size,bptt_len=50,repeat=False,shuffle=True)
# it=iter(trainiter)
# batch = next(it)
```

Build a new RNN model.

```python
class RNNmodel(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size):
        super(RNNmodel,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size)     # 通过设置batch_first = True ，输入可以是 batch_size*seq_length其他不用变
        self.decoder = nn.Linear(hidden_size,vocab_size)
        self.hidden_size = hidden_size
 
    def forward(self,text,hidden):
        emb = self.embed(text)       # text: seq_length*batch_size
        output,hidden = self.lstm(emb,hidden) # output: seq_length*batch_size*hidden_size   hidden: (1*batch_size*hidden_size,1*batch_size*hidden_size  )
        decoded = self.decoder(output.view(-1,output.size(-1)))
        decoded = decoded.view(output.size(0),output.size(1),decoded.size(1))  # decoded: seq_length * batch_size * vocab_size
        return decoded, hidden
 
 
    def init_hidden(self,bsize,requires_grad=True):
        weight = next(self.parameters())
        return (weight.new_zeros((1,bsize,self.hidden_size),requires_grad=requires_grad),
                weight.new_zeros((1,bsize,self.hidden_size),requires_grad=requires_grad))
```

Train RNN and predict on test data.

```python
model = RNNmodel(len(TEXT.vocab),embedding_size,hidden_size)
optimizer=torch.optim.Adam(model.parameters(),lr=learn_rate)
loss_fn = nn.CrossEntropyLoss()
 
 
def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
 
 
def evaluate(model,data):
    model.eval()
    total_loss = 0.
    total_count = 0.
    it = iter(data)
    with torch.no_grad():
        hidden = model.init_hidden(batch_size,requires_grad=False)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            hidden = repackage_hidden(hidden)
 
            output, hidden = model(data, hidden)
            loss = loss_fn(output.view(-1, len(TEXT.vocab)), target.view(-1))
            total_loss = loss.item()*np.multiply(*data.size())
            total_count = np.multiply(*data.size())
    loss = total_loss/total_count
    model.train()
    return loss
 
val_losses = []
GRAD_CLIP =5.0
for epoch in range(2):
    model.train()
    it = iter(train_iter)
    hidden = model.init_hidden(batch_size)
    for i, batch in enumerate(it):
        data,target = batch.text,batch.target
        hidden = repackage_hidden(hidden)
        output,hidden = model(data,hidden)
        loss = loss_fn(output.view(-1, len(TEXT.vocab)),target.view(-1))  # batch_size * target_class_dim    batch_size
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)        # 优化方法
        loss.backward()
        optimizer.step()
        schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)         # 调整学习率来优化，0.5即降一半
        if i % 100 == 0:
            print('loss',loss.item())
        if i % 1000 == 0:
            val_loss = evaluate(model,val_iter)
            if len(val_losses)==0 or val_loss < min(val_losses):
                torch.save(model.state_dict(),'lm.pth')
                print('best model saved to lm.pth')
            else:
                schedule.step()
                optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
            val_losses.append(val_loss)
```

### Results

#### Model Parameters

```python
OrderedDict([('embed.weight',
              tensor([[ 1.0244,  0.6886,  1.0996,  ...,  0.7399, -1.5735, -1.2679],
                      [-1.3549, -0.3843, -0.0618,  ...,  0.0501,  0.0907,  0.3998],
                      [-1.1766, -2.1427,  1.0838,  ..., -0.5362,  1.1429,  1.1455],
                      ...,
                      [ 0.9622,  0.4979, -0.2692,  ..., -0.2930,  0.3888, -0.4008],
                      [ 2.0822, -0.2946,  3.1807,  ..., -0.2909, -1.3140,  0.2740],
                      [-0.2806, -0.4378, -1.5259,  ...,  0.6442,  0.7118,  0.4313]])),
             ('lstm.weight_ih_l0',
              tensor([[ 0.0493, -0.0120,  0.0042,  ...,  0.0691, -0.0893, -0.0576],
                      [-0.0631,  0.0954,  0.0700,  ..., -0.0719, -0.0832, -0.0846],
                      [-0.0865,  0.0666,  0.0163,  ..., -0.0831,  0.0840, -0.0412],
                      ...,
                      [ 0.0771,  0.0994,  0.0831,  ..., -0.0415,  0.0637, -0.0982],
                      [ 0.0084,  0.0736,  0.0819,  ..., -0.0168, -0.0050, -0.0267],
                      [-0.0400, -0.0251, -0.0182,  ..., -0.0186,  0.0062, -0.0597]])),
             ('lstm.weight_hh_l0',
              tensor([[ 0.0614,  0.0650, -0.0627,  ...,  0.0460, -0.1007, -0.0334],
                      [ 0.0739, -0.0581, -0.0153,  ..., -0.0271, -0.0163,  0.0480],
                      [ 0.0483,  0.0747,  0.0758,  ..., -0.0976,  0.0818, -0.0708],
                      ...,
                      [-0.1004, -0.0455,  0.0013,  ...,  0.0232,  0.0174, -0.0453],
                      [ 0.0878,  0.0277, -0.0279,  ..., -0.0308, -0.0170,  0.0258],
                      [ 0.0701, -0.0790,  0.0007,  ..., -0.0044,  0.0134, -0.0092]])),
             ('lstm.bias_ih_l0',
              tensor([ 1.3788e-02, -6.7370e-02, -2.4378e-02,  2.1658e-02,  7.5325e-02,
                      -3.1794e-02,  4.9366e-02,  2.1986e-02, -7.3339e-02, -3.3048e-02,
                      -3.5299e-02,  5.5914e-02, -9.5530e-02,  3.8095e-02, -9.2605e-03,
                      -3.5058e-02, -1.7872e-02,  2.9417e-02, -9.2673e-02,  2.0403e-02,
                       7.4280e-02,  7.6273e-02, -5.4189e-02,  7.9059e-02, -4.9155e-03,
                      -8.5133e-02,  7.2287e-02,  5.5351e-02,  6.9964e-02, -8.9710e-02,
                      -2.4062e-02,  1.6417e-03,  3.9593e-02, -7.1047e-02,  5.1596e-02,
                       9.7804e-03,  9.6118e-02,  9.3320e-02,  7.2845e-02, -2.2085e-02,
                       5.1085e-02,  8.0564e-02, -7.4941e-02, -7.3707e-02,  9.6328e-02,
                       2.2575e-02,  7.4014e-02, -5.7252e-02,  3.5155e-02, -3.0350e-02,
                       6.4458e-02,  6.0375e-02,  2.6508e-02, -2.1357e-02, -1.9226e-02,
                      -7.9563e-02, -3.0160e-02,  2.3216e-04, -6.1966e-02,  3.1087e-02,
                       3.3720e-02, -9.4892e-02, -8.1161e-02,  3.8857e-02, -2.4667e-03,
                       3.0770e-02, -8.0868e-02,  8.9641e-03, -6.9890e-02, -3.8234e-02,
                      -2.1690e-02, -4.9843e-02,  9.3746e-02, -3.2636e-03,  9.8620e-02,
                      -5.3164e-02,  5.0785e-03,  6.5226e-02, -9.4536e-02, -6.5368e-02,
                      -1.7029e-03, -7.8001e-02,  6.5427e-02,  4.8509e-02, -9.3469e-02,
                      -3.6165e-02,  6.0035e-02,  1.7428e-02, -2.1437e-02, -7.1697e-02,
                      -8.5672e-03, -7.0152e-02,  9.6242e-02, -4.4290e-02,  9.2110e-02,
                       9.0298e-02, -7.1285e-02, -1.2875e-02, -9.5336e-03, -7.4640e-02,
                      -1.4640e-02, -3.3260e-02,  5.4105e-02,  7.1760e-02,  8.7481e-02,
                       3.9027e-02,  8.7718e-02,  5.8399e-02,  1.8424e-02, -2.9009e-02,
                       1.2583e-02, -4.2179e-02,  6.5607e-02, -7.9110e-02, -9.1230e-02,
                      -7.6649e-02, -5.2135e-02,  2.2974e-02,  6.1348e-02,  5.0820e-02,
                       6.7977e-02, -3.6404e-02, -5.3094e-02, -9.6674e-02,  2.1544e-02,
                      -6.3066e-02, -4.6106e-02, -8.0219e-02,  7.3068e-02,  6.9733e-02,
                      -7.1811e-02,  2.8855e-03, -9.1674e-02,  1.7734e-03, -1.6444e-02,
                      -5.3966e-02,  2.5775e-02,  3.9203e-02,  9.3501e-02,  8.2208e-02,
                       5.6137e-04, -5.0701e-02, -9.7410e-02,  9.1561e-02, -6.5155e-02,
                      -9.3311e-02,  8.3075e-02, -5.9749e-02, -3.8264e-02, -6.7996e-02,
                       7.2443e-02,  9.8333e-02, -4.7138e-02,  9.1317e-03,  8.7531e-02,
                      -2.6475e-02,  1.3988e-03,  6.6071e-02, -5.3868e-02, -8.7029e-02,
                      -7.9623e-02, -6.3007e-02, -9.3534e-02, -6.8808e-04,  5.6065e-02,
                      -3.7144e-02,  5.9982e-02, -8.4392e-02, -3.9850e-02,  6.2749e-02,
                      -6.3193e-02,  6.3041e-02,  6.3065e-02, -3.1157e-02,  3.7535e-02,
                      -7.8122e-02,  7.1338e-02,  7.5557e-02,  9.4477e-02, -5.5938e-02,
                      -7.9690e-02, -2.8669e-02, -6.7646e-02,  8.9021e-02, -4.1347e-03,
                      -4.2232e-02,  6.4971e-02,  7.1347e-02,  4.9675e-02,  4.7689e-02,
                      -8.4380e-02, -9.5990e-02,  8.8916e-02, -9.3582e-02,  4.2401e-02,
                       2.0510e-02,  5.7748e-03,  8.6076e-02, -5.9546e-02,  2.1120e-02,
                       1.3249e-02, -8.6261e-02,  4.8170e-03,  7.7395e-02,  5.2682e-02,
                      -3.0578e-02,  9.3871e-02,  4.6140e-02,  3.7110e-02, -9.2300e-02,
                      -7.6118e-02, -4.2556e-03,  6.7192e-02, -4.3463e-02, -9.2280e-02,
                       2.0834e-02,  5.1566e-02,  4.7549e-02, -6.0848e-02,  9.5194e-02,
                       4.4154e-02,  6.6988e-02, -2.9484e-03,  6.8950e-02, -4.8158e-02,
                      -9.3083e-02, -2.7917e-02,  5.0569e-02,  6.9619e-02, -8.8710e-02,
                      -8.3468e-02, -4.9874e-02, -7.5884e-02, -7.2653e-02,  5.9151e-02,
                       9.5441e-03, -4.1861e-05,  1.8112e-02,  2.4090e-02, -6.8458e-02,
                      -4.1312e-02,  6.1826e-02, -8.2238e-03,  7.0733e-02,  9.5937e-02,
                       7.6407e-02,  8.8279e-02,  5.2441e-02, -1.8540e-02, -7.7596e-03,
                       8.4793e-02, -3.0815e-02,  7.8824e-02, -9.1632e-02, -6.0900e-02,
                       5.6014e-03,  6.1172e-02, -6.2031e-02,  6.0376e-02,  4.4096e-02,
                       6.5675e-02,  5.9841e-02,  4.4070e-02, -5.7940e-02, -6.6477e-02,
                       7.1452e-02,  1.6031e-02, -4.0916e-02, -9.7461e-03,  8.3183e-02,
                       9.5330e-02,  8.0226e-02,  2.0938e-02,  2.8851e-02, -2.0074e-02,
                      -3.7864e-02, -2.9707e-02,  3.5440e-02, -6.6685e-02, -2.4256e-02,
                      -4.0834e-02,  2.1287e-02,  4.2744e-02,  6.9908e-02,  6.9609e-02,
                      -4.3244e-02, -9.4745e-02, -8.3614e-02,  4.9706e-02, -3.3530e-02,
                      -6.5122e-03, -4.9816e-02,  6.5326e-02, -7.6172e-02, -5.9342e-02,
                       1.0106e-02,  2.4899e-02, -4.6940e-02, -7.0142e-02,  6.6212e-02,
                       1.3555e-02, -1.5921e-02,  7.0184e-02, -2.5501e-02, -1.3986e-02,
                      -9.7160e-02,  9.0499e-02, -5.0260e-02, -8.5275e-02,  1.1974e-03,
                      -7.9326e-02,  5.3141e-02,  7.1196e-02, -2.8526e-02, -9.4598e-02,
                       2.9482e-02, -1.3412e-03, -8.9683e-02,  6.3312e-02, -7.9388e-02,
                       1.9890e-03, -4.4917e-02,  9.9118e-02,  6.9068e-02, -1.0217e-01,
                       9.3654e-02,  8.1320e-02, -2.9897e-02,  9.1756e-02, -6.7499e-04,
                       3.7886e-02, -8.1684e-03,  5.6087e-02, -1.6659e-02, -3.6985e-03,
                      -1.4717e-02, -9.0360e-02,  7.4773e-02,  3.1879e-02, -9.7654e-03,
                      -6.5927e-03, -5.3398e-02, -2.3044e-02, -7.8609e-02, -1.7685e-02,
                       1.5864e-02, -6.3587e-04, -9.4018e-02,  7.3376e-02, -4.8756e-03,
                      -9.2938e-03, -9.4781e-02, -9.9160e-03,  9.6673e-02,  9.6132e-02,
                      -8.7816e-02,  1.6767e-02, -2.5712e-02, -9.6574e-02,  7.5403e-02,
                       1.6311e-02, -4.7201e-03, -3.1617e-02,  9.2977e-02,  4.4335e-02,
                       3.0157e-02,  1.4782e-02, -7.9044e-02, -9.7355e-02,  4.6654e-02,
                      -4.0249e-03, -8.3627e-02, -1.2718e-03,  2.3964e-02, -5.0466e-02,
                       7.4437e-02,  9.6537e-03,  2.1528e-02, -8.6539e-02, -4.3398e-02,
                       6.2591e-03, -5.4261e-02, -7.4150e-02,  5.5501e-03, -4.0183e-02,
                      -3.7430e-02, -1.9676e-02, -4.7791e-02, -4.9978e-02,  3.8457e-02,
                       7.7709e-02,  1.1133e-02, -1.0107e-01,  3.4488e-02,  1.4842e-02,
                       8.8079e-03, -3.1985e-02, -8.1446e-04, -3.1342e-02,  9.5025e-02])),
             ('lstm.bias_hh_l0',
              tensor([-6.4809e-02,  1.8917e-03, -8.1165e-02, -5.1213e-03, -1.8657e-02,
                      -6.5591e-02, -5.4456e-02,  6.0764e-02, -8.7199e-02, -4.2966e-03,
                       8.0290e-02,  4.0503e-02,  8.4251e-02,  3.0865e-02, -4.5851e-02,
                      -3.9795e-02,  6.9312e-02,  4.8740e-02, -1.3554e-03,  4.9733e-02,
                       1.0226e-01,  9.7489e-02,  7.4796e-02, -7.6395e-02, -3.5202e-02,
                       1.0832e-02, -3.1706e-02, -7.1993e-02, -4.5254e-02, -6.6611e-02,
                       6.2179e-02, -9.5345e-02, -5.3077e-03,  7.6380e-02, -5.4308e-02,
                       5.7003e-02,  1.1419e-02, -2.5661e-02,  3.3799e-02,  5.8142e-03,
                      -1.0105e-01, -8.7298e-02, -7.0414e-02, -7.9292e-02, -9.4219e-02,
                       1.0392e-02,  5.5646e-02, -8.5274e-02,  1.0042e-01,  2.3386e-02,
                       3.3594e-02,  4.5745e-03, -8.8812e-02,  4.0786e-02,  6.5321e-02,
                      -3.6373e-02, -5.8188e-02, -8.6768e-02,  7.9013e-04,  3.6507e-02,
                       6.4091e-02, -6.5609e-02, -3.8025e-02,  4.8950e-02,  7.7492e-02,
                      -2.2432e-02, -2.1098e-03, -1.1939e-02, -1.1075e-02, -8.1749e-02,
                       8.7306e-02, -1.3046e-02, -5.9730e-02,  8.1672e-02, -1.8941e-02,
                       5.2889e-02, -3.1124e-02,  3.5400e-05, -5.0267e-02,  3.9641e-02,
                      -3.5796e-02, -3.9984e-03,  9.8349e-02,  8.1354e-02,  7.4268e-02,
                       2.6936e-02,  9.6160e-02, -1.8043e-02,  2.3950e-02, -9.0921e-02,
                       7.3040e-02,  5.5895e-02, -3.5814e-02, -9.4142e-02,  2.1261e-02,
                       9.8674e-02, -1.3979e-02, -1.3877e-02, -7.5998e-03,  1.4986e-03,
                       9.4020e-02, -4.4457e-02,  3.5987e-03, -1.7021e-02,  2.2941e-03,
                      -1.4078e-03, -8.7305e-02, -3.5837e-02,  7.3415e-02,  8.2531e-04,
                      -7.4873e-02,  7.3322e-02, -1.0211e-01,  2.3370e-02,  4.7746e-02,
                      -1.5199e-02, -3.7278e-02,  7.0539e-02, -7.8057e-02,  3.5841e-02,
                       8.8652e-02, -2.3997e-02, -6.7488e-03, -1.0031e-01,  5.9037e-02,
                       6.1076e-02, -4.6819e-02,  3.6003e-02, -2.3333e-02,  2.2680e-02,
                       5.4227e-02,  9.1412e-02,  6.9485e-02, -1.8474e-02,  4.7419e-02,
                       8.2954e-02, -4.9623e-02,  1.4070e-02,  7.6278e-02, -2.9223e-02,
                       2.1148e-02,  3.0287e-02, -2.1395e-02,  7.9799e-02, -1.0629e-02,
                       5.0535e-02, -7.7333e-02, -2.5352e-02, -2.9528e-02, -4.9627e-02,
                      -4.1740e-02, -4.8594e-02,  2.2263e-02,  5.3948e-02,  5.9028e-02,
                      -2.0670e-02,  8.1681e-02,  8.1118e-02, -8.5905e-03,  8.3024e-03,
                       1.8590e-02, -6.1217e-02, -2.7951e-02, -7.7733e-02,  1.0671e-02,
                      -5.1182e-02, -2.7289e-03, -6.3929e-02, -3.5130e-02, -6.1638e-02,
                       3.2618e-02, -1.5771e-02, -4.5467e-02, -5.3470e-02,  4.8042e-02,
                      -1.9126e-02, -2.6938e-02,  3.2224e-02, -1.2924e-02, -5.9819e-02,
                      -5.9864e-02,  9.3304e-02, -6.2491e-02,  6.9224e-02,  5.5625e-02,
                       8.2447e-02, -1.4472e-03,  7.2162e-02,  9.6516e-02, -5.3087e-02,
                      -2.1401e-02, -7.6994e-02,  1.2531e-02,  9.4152e-02, -3.6543e-02,
                      -6.1557e-02,  9.8112e-02,  7.7609e-02, -6.8473e-02, -6.5059e-02,
                      -9.2580e-02, -5.9610e-02, -4.2702e-02, -4.8590e-02,  6.7875e-02,
                       6.7299e-02,  1.0370e-01, -3.6853e-02, -1.8854e-02,  6.0087e-02,
                       9.3330e-02,  6.7301e-02, -9.6446e-02, -1.8273e-02,  8.5345e-02,
                       2.2605e-02,  4.9688e-02,  5.6550e-02,  7.3799e-02,  9.9649e-02,
                       1.6829e-02,  1.9305e-02,  8.1421e-03,  3.8440e-02,  9.4080e-02,
                      -1.0802e-02, -1.4064e-02,  3.6923e-02,  5.3006e-02, -6.9352e-02,
                       5.2217e-02, -9.1281e-02, -5.8210e-02,  6.3397e-02,  6.6053e-02,
                       5.9548e-03,  2.3775e-02,  6.5225e-02, -7.9916e-02, -2.6486e-02,
                       3.9984e-02,  1.0648e-02,  1.3431e-02, -8.6885e-02,  5.0814e-02,
                      -9.2493e-02,  3.0833e-02, -1.9677e-02, -5.8226e-02,  3.0804e-02,
                       4.5188e-02, -8.5001e-02, -2.2885e-02,  7.7884e-02, -7.0132e-02,
                      -9.5271e-02,  9.1981e-02, -2.0531e-02,  9.3708e-02, -8.0425e-03,
                      -4.0516e-02,  3.3180e-02, -9.7989e-02, -9.2074e-02,  8.5885e-02,
                      -5.7256e-02, -4.0004e-02, -6.7940e-02,  7.6134e-02,  2.6924e-03,
                      -3.5118e-02,  2.8794e-02,  4.4097e-03,  7.0824e-02,  3.1173e-02,
                       8.0696e-02,  1.3470e-02, -6.4002e-02,  9.6470e-04,  4.2573e-02,
                       9.7038e-02, -4.2354e-02, -4.5885e-02, -8.8559e-02, -8.3395e-02,
                      -4.5924e-02,  1.5135e-03,  2.2753e-02, -3.1785e-02, -2.0010e-02,
                      -3.3131e-02,  9.0326e-02,  2.4814e-02,  1.3690e-02, -6.1992e-02,
                       2.7289e-02,  7.0605e-02, -3.3027e-02, -9.7549e-03,  8.7704e-02,
                       5.0281e-02,  3.2466e-02, -6.6275e-02,  5.2536e-03, -5.6842e-02,
                      -3.1853e-02,  7.8048e-02,  7.0614e-03,  5.4006e-02, -4.6424e-02,
                      -7.6265e-02,  4.1834e-03,  3.1228e-02,  2.6981e-02, -8.3496e-02,
                       2.8215e-02,  5.3443e-03, -1.1344e-03,  2.0663e-02, -9.2703e-02,
                      -7.8152e-02,  9.6353e-02, -2.8872e-02,  6.7817e-02, -1.6957e-02,
                       7.5252e-02, -3.6801e-02, -1.0065e-01, -8.6992e-02,  5.2059e-02,
                      -9.1416e-02,  4.8202e-02, -3.8636e-02, -4.6564e-02, -5.1037e-02,
                      -1.3984e-02,  4.7509e-02,  9.1151e-02,  3.0422e-02,  5.1338e-03,
                      -6.1686e-02, -6.0618e-02, -9.5426e-02, -9.5806e-02, -7.4450e-02,
                      -4.9590e-02,  8.8050e-02, -7.8876e-02,  8.3046e-02,  4.0854e-03,
                      -9.5182e-02,  2.7865e-03, -6.1203e-02, -8.7769e-02,  8.6926e-02,
                      -7.3662e-02,  4.2146e-02,  3.0260e-02, -3.2655e-02,  3.2412e-02,
                      -2.2421e-04, -7.7164e-02,  4.1686e-02,  9.6120e-02, -3.7624e-02,
                       4.5112e-02,  6.4656e-02,  4.2612e-02,  6.4842e-02, -8.6971e-03,
                      -1.5846e-02, -5.6664e-02,  7.5695e-03, -6.8523e-02,  7.8330e-02,
                      -7.6198e-02,  7.9196e-02, -2.7029e-02,  6.9065e-02,  3.0245e-02,
                       4.2388e-02,  5.0541e-02,  3.8796e-02, -8.7031e-02,  4.8825e-02,
                      -9.5418e-02,  2.8625e-02, -5.2599e-02, -6.6369e-02,  5.3767e-02,
                      -3.4216e-02, -5.9605e-02, -5.1880e-02, -3.3874e-02,  9.0366e-02,
                      -8.6266e-02, -2.4110e-02, -3.8682e-02, -1.6562e-02,  9.9872e-03])),
             ('decoder.weight',
              tensor([[-0.0187,  0.0776,  0.0207,  ..., -0.0654, -0.0540, -0.0227],
                      [-0.0650,  0.0435,  0.0734,  ..., -0.0747, -0.0888, -0.0269],
                      [ 0.0122, -0.0111,  0.0963,  ..., -0.0399, -0.0329,  0.0524],
                      ...,
                      [ 0.0621,  0.0239, -0.0023,  ..., -0.0460, -0.0013, -0.0832],
                      [-0.0055, -0.0851,  0.0323,  ...,  0.0033, -0.0988, -0.0630],
                      [-0.0878, -0.0664, -0.0976,  ..., -0.0173,  0.0509,  0.0062]])),
             ('decoder.bias',
              tensor([-0.0144, -0.0434, -0.0882,  ...,  0.0938,  0.0907, -0.0327]))])
```

#### Perplexity

```python
perplexity:  930.8939872232444
```

Perplexity of RNNLM goes larger because the train data size is much smaller.

## 3. Language Model with Generative Pre-training(BERT)

I use BERT as a LM with Generative Pre-training for prediction here. It's based on transformer and a PLM with good performance. Model Structure is shown as below:

![截屏2022-11-09 下午4.42.14](/Users/renee/Desktop/截屏2022-11-09 下午4.42.14.png)

### Code

```python
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM
```

```python
def computeppl(sen):
    with torch.no_grad():
        model = BertForMaskedLM.from_pretrained('hfl/chinese-bert-wwm-ext')
        model.eval()
        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
        sentence = sen
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        sen_len = len(tokenize_input)
        sentence_loss = 0.

        for i, word in enumerate(tokenize_input):
            # add mask to i-th character of the sentence
            tokenize_input[i] = '[MASK]'
            mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])

            output = model(mask_input)

            prediction_scores = output[0]
            softmax = nn.Softmax(dim=0)
            ps = softmax(prediction_scores[0, i]).log()
            word_loss = ps[tensor_input[0, i]]
            sentence_loss += word_loss.item()

            tokenize_input[i] = word
        ppl = np.exp(-sentence_loss/sen_len)
    return(ppl)
```

```python
import re
def getText(fileName):
    all_text = ""
    file_object = open(fileName, 'r',encoding = 'utf-8')
    try:
        all_text += file_object.read()
    finally:
        file_object.close()
    print(all_text)
    re_patten = '[^\u4e00-\u9fa5？。，！]'
    all_text = re.sub(re_patten, "", all_text.strip())
    #reChinese = re.compile('/[(\u4e000-\u9fff)(A-Za-z0-9)(，,（）【】{}！\-!)]+/g ') 
    #all_text = reChinese.findall(all_text) 
    #all_text = all_text.split()
    #for item in all_text:
        #print(item)
    return all_text
  
ppltotal=0
count=0
for i in a[0:1169]:  
    print(i)
    ppltotal+=computeppl(i)
    count+=1
    print(count)
    print(ppltotal)
```

### Results

```
Perplexity : 3776088.8705747942
```

Perplexity here is much larger than RNNLM and NNLM, because perplexities of some sentences explode. Performance for predition is unstable.

 
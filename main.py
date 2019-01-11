import torch
import gensim
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import time
import numpy as np
from model import  textCNN_Kim
from torch.utils.data import TensorDataset
import random

MODEL_PATH = "models/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5"
POS_PATH = "seg_pos.txt"
SEG_PATH = "seg_neg.txt"

def vectorLoad(model_path):
    time_start = time.time()
    wordVec = gensim.models.KeyedVectors.load_word2vec_format(model_path,binary=False)
    print('model loadcost '+str(time.time()-time_start))
    return wordVec

def line_to_vector(line):
    sentence_emb = []
    for element in line.split(','):
        word = element.split('/')[0].strip()
        try:
            sentence_emb.append(word_Vec[word])
        except:
            pass
    return sentence_emb

def load_data(path,label):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        #i = 0
        total_emb = []
        for line in lines:
            # i += 1
            # if i < 30:
                #print(line)
            sentenc_emb = line_to_vector(line)
            #print(sentenc_emb)
            total_emb.append(sentenc_emb)
        len_ = len(total_emb)
        if label == 1:
            labels = np.ones(len_)
        else:
            labels = np.zeros(len_)
        return list(total_emb),list(labels)

word_Vec = vectorLoad(MODEL_PATH)
pos_datas,pos_labels = load_data(POS_PATH,1)
# print(pos_datas)
# print(pos_labels)
seg_datas,seg_labels = load_data(SEG_PATH,0)


pos_datas.extend(seg_datas)
pos_labels.extend(seg_labels)

all_datas = pos_datas
all_labels = pos_labels
#all_datas = np.concatenate(pos_datas , seg_datas)
#all_labels = np.concatenate(pos_labels , seg_labels)
print("all data length:{}".format(len(all_datas)))

data = list(zip(all_datas,all_labels))
random.shuffle(data)
la = int(len(data)*0.8)

#print(np.array(data[:la]))
train_data = data[:la]
batch_size = 3
train_data_x_batch = []
train_data_y_batch = []

#train_data_x =[train_data for i,train_data in enumerate(train_data)]


train_data_x = []
train_data_y = []
for i,(x,y) in enumerate(train_data):
    train_data_x.append(x)
    train_data_y.append(y)
    if i != 0 and i % batch_size == 0:
        train_max_len = max((len(l) for l in train_data_x))
        train_data_x = [np.vstack((l, np.zeros(((train_max_len - len(l)), 300)))) for l in train_data_x]
        train_data_x_batch.append(train_data_x)
        train_data_y_batch.append(train_data_y)
        train_data_x = []
        train_data_y = []
print("train data size:{}".format(len(train_data_x_batch)))

train_data_x_b = torch.FloatTensor(train_data_x_batch)
train_data_y_b = torch.LongTensor(train_data_y_batch)
#train_data_x = torch.FloatTensor(train_data_x)
print("-----")

# test_data = data[la:]
# test_data_x = []
# test_data_y = []
# for i,(x,y) in enumerate(test_data):
#     test_data_x.append(x)
#     test_data_y.append(y)
# print("test data size:{}".format(len(test_data_x)))
# test_max_len = max((len(l) for l in test_data_x))
# print("test_max_len：{}".format(test_max_len))
# #test_data_x = np.array([np.vstack((l, np.zeros(((test_max_len - len(l)), 300)))) for l in test_data_x]).astype(np.float64)
# test_data_x = torch.FloatTensor(test_data_x)
# test_data_y = torch.LongTensor(test_data_y)


net = textCNN_Kim(1,300,2,100,[2,3,4])
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
net.cuda()
print(net)
optimizer = torch.optim.Adam(net.parameters(),lr=0.01)
loss_function = torch.nn.CrossEntropyLoss()
## MultiLabelSoftMarginLoss

# torch_dataset = TensorDataset(train_data_x,train_data_y)
# train_loader = Data.DataLoader(dataset=torch_dataset,batch_size=50,shuffle=True)

EPOCH = 1
for epoch in range(EPOCH):
    for index,item in train_data_x_b:
        x = item.cuda()
        y = train_data_y_b[index].cuda


    # for step,(x,y) in enumerate(train_loader):
    #     x.cuda()
    #     y.cuda()
    #     out = net(x)
    #     # print(out)
    #     # print(out.size())
    #     loss = loss_function(out,y).cuda()
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     if step % 2 == 0:
    #         net.eval()
    #         predict = net(test_data_x.cuda())
    #         #print(predict)
    #         pred_y = torch.max(predict,1)[1].data.numpy()
    #         accuracy = float((pred_y==test_data_y.data.numpy()).astype(int).sum())/float(len(test_data_y))
    #         print("epoch:%f,step:%f,loss:%.4f,accuracy:%.4f" % (epoch,step,loss.data.numpy(),accuracy))
    #         net.train()







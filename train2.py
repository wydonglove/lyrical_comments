import gensim
import time
import random
import numpy as np
import torch
from model import textCNN_Kim
import torch.nn.functional as F
# import preprocess.vectorization as vectorization
import sys
import os
import gc
from torch.autograd import Variable as V

def vectorLoad(model_path):
    time_start = time.time()
    wordVec = gensim.models.KeyedVectors.load_word2vec_format(model_path,binary=False)
    vecs_ = []
    missings_ =[]

    print('model loadcost '+str(time.time()-time_start))
    return wordVec


def randomChoice1(l):
    return l[random.randint(0, len(l) - 1)]
def randomChoice(len):
    return random.randint(0, len - 1)

# def loadCorpus(seg_path):
#     docs =[]
#     docsObj ={}
#     docs_id =0
#     with open(seg_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         for line in lines:
#             docs.append(line)
#     random.shuffle(docs)
#     for doc in docs:
#         docsObj[docs_id]=doc
#         docs_id+=1
#     return docsObj

def corpusMerge(posCorpus,negCorpus):
    corpus_id =[]
    last_id = None
    lineDict = {}
    labelDict = {}
    for id in posCorpus.keys():
        corpus_id.append(id)
        lineDict[id] = posCorpus[id]
        labelDict[id] = 1
        last_id = id
    for id in negCorpus.keys():
        corpus_id.append(id+last_id)
        lineDict[int(id)+int(last_id)] = negCorpus[id]
        labelDict[int(id)+int(last_id)] = 0

    return corpus_id,lineDict,labelDict

def loadPostive(seg_path):
    docs =[]
    docsObj ={}
    docs_id =0
    with open(seg_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            docs.append(line)
    # random.shuffle(docs)
    for doc in docs:
        docsObj[docs_id]=doc
        docs_id+=1
    return docsObj

def loadNegtive(seg_path):
    docs =[]
    docsObj ={}
    docs_id =0
    with open(seg_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            docs.append(line)
    # random.shuffle(docs)
    for doc in docs:
        docsObj[docs_id]=doc
        docs_id+=1
    return docsObj

def line2Tensor(line):

    sentence_emb = []
    for element in line.split(',') :
        word = element.split('/')[0].strip()
        try:
            sentence_emb.append(wordVec[word])
        except:
            pass
    return np.array(sentence_emb)

def save(model, save_dir, save_prefix,epoch):
    print('start save..')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_epoch_{}.pt'.format(save_prefix,epoch)
    torch.save(model.state_dict(), save_path)

def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


# model_path ='models/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
#model_path ='models/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
model_path='models/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5'
#model_path='/home/yangyanqi/models/sgns.merge.bigram'
pos_segPath ='seg_pos.txt'
neg_segPath ='seg_neg.txt'

wordVec = vectorLoad(model_path)


posCorpus = loadPostive(pos_segPath)
print(posCorpus)
negCorpus = loadNegtive(neg_segPath)

# corpus_id : [id],corpus : {key:id,value:line},labels: {key:id,value:0 }
corpus_id, corpus,labels = corpusMerge(posCorpus, negCorpus)
input_dim =300
log_interval = 1
save_interval=100
steps = 0
best_acc = 0
last_step = 0
save_dir = "weights/"
epoch = 50
batch_size = 32
batch_size_test = 8


model = textCNN_Kim(input_width=1,input_dim=input_dim,class_num=2,kernel_num=200,kernel_sizes=[2,3,4,5,6,7])
#model =model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



train_test_rate = 0.8
sess = 'train'
random.seed(0)
random.shuffle(corpus_id)

train_id, test_id = corpus_id[0:int(len(corpus_id) * train_test_rate)], corpus_id[int(len(corpus_id) * train_test_rate):]
if sess == 'train':
    model.train()
    for epoch in range(1, epoch+1):
        # train test shuffle
        # random.shuffle(corpus_id)
        random.shuffle(train_id)
        random.shuffle(test_id)

        train_batches_id = []
        test_batches_id = []
        batch_id_tmp = []
        for i, e in enumerate(train_id):
            batch_id_tmp.append(e)
            if (i + 1) % batch_size == 0:
                train_batches_id.append(batch_id_tmp)
                batch_id_tmp = []

        batch_id_tmp = []
        for i, e in enumerate(test_id):
            batch_id_tmp.append(e)
            if (i + 1) % batch_size_test == 0:
                test_batches_id.append(batch_id_tmp)
                batch_id_tmp = []
        test_batches_id.append(test_id)


        batch_num = np.shape(train_batches_id)[0]
        i =0
        for batches in train_batches_id:
            lineTensors = []
            labels_ = []
            max_line_length = 0
            batch = []
            for id in batches:
                line =corpus[id]
                label = labels[id]
                labels_.append(label)
                line = line[1:-1].strip()
                lineTensor = np.array(line2Tensor(line))
                length = np.shape(lineTensor)[0]
                if length>max_line_length:
                    max_line_length = length
                lineTensors.append(lineTensor)

            max_len = max((len(l) for l in lineTensors))
            inputs = np.array([np.vstack((l, np.zeros(((max_len - len(l)), 300)))) for l in lineTensors]).astype(np.float64)
            targets = np.array(labels_)
            targets = torch.Tensor(targets).long()
            targets=V(targets.cuda())
            inputs = torch.Tensor(inputs)
            optimizer.zero_grad()
            logit = model(inputs)


            loss = F.cross_entropy(logit, targets)
            loss.backward()
            optimizer.step()

            steps += 1
            i+=1
            if steps % log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(targets.size()).data == targets.data).sum().cpu()
                accuracy = 100.0 * corrects/batch_size
                print('\repoch[{}] - Batch[{}] - loss: {:.6f}  acc: {:.4f}%()'.format(
                                                                            epoch,
                                                                            steps,
                                                                             loss.item(),
                                                                             accuracy
                                                                             # corrects
                                                                                ))

            # if steps % test_interval == 0:
            #     dev_acc = eval(dev_iter, model, args)
            #     if dev_acc > best_acc:
            #         best_acc = dev_acc
            #         last_step = steps
            #         if args.save_best:
            #             save(model, args.save_dir, 'best', steps)
            #     else:
            #         if steps - last_step >= args.early_stop:
            #             print('early stop by {} steps.'.format(args.early_stop))
            # elif steps % save_interval == 0:
            #     save(model, save_dir, 'snapshot', steps)
            # if batch_num==i:
            #     corrects = (torch.max(logit, 1)[1].view(targets.size()).data == targets.data).sum().cpu()
            #     accuracy = 100.0 * corrects / batch_size
            #     print('\repoch[{}] - loss: {:.6f}  acc: {:.4f}%()'.format(
            #         epoch,
            #         loss.data.cpu().numpy(),
            #         accuracy
            #         # corrects
            #     ))

            # if batch_num==i:
        save(model, save_dir, 'model_k200_234567_',epoch)
        steps = 0
            # sess='eval'
if sess == 'eval':
    batch_id_tmp = []
    test_batches_id =[]
    random.shuffle(test_id)

    for i, e in enumerate(test_id):
        batch_id_tmp.append(e)
        if (i + 1) % batch_size_test == 0:
            test_batches_id.append(batch_id_tmp)
            batch_id_tmp = []
    # test_batches_id.append(test_id)

    print('eval')
    model.load_state_dict(torch.load('weights/model_k200_epoch_50.pt'))
    model.eval()
    corrects, avg_loss = 0, 0
    tp,fn,tn,fp = 0,0,0,0
    for batches in test_batches_id:
        lineTensors = []
        labels_ = []
        max_line_length = 0
        batch = []
        for id in batches:
            line =corpus[id]
            label = labels[id]
            labels_.append(label)
            line = line[1:-1].strip()
            lineTensor = np.array(line2Tensor(line))
            length = np.shape(lineTensor)[0]
            if length>max_line_length:
                max_line_length = length
            lineTensors.append(lineTensor)

        max_len = max((len(l) for l in lineTensors))
        inputs = np.array([np.vstack((l, np.zeros(((max_len - len(l)), 300)))) for l in lineTensors]).astype(np.float64)
        targets = np.array(labels_)
        targets = torch.Tensor(targets).long()
        targets=V(targets.cuda())
        inputs = torch.Tensor(inputs)
        optimizer.zero_grad()
        logit = model(inputs)

        loss = F.cross_entropy(logit, targets, size_average=False)

        # avg_loss +=  loss.data.cpu().numpy(),
        avg_loss +=  loss.item()
        # avg_loss +=  loss.item(),

        # corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        corrects += (torch.max(logit, 1)[1].view(targets.size()).data == targets.data).sum().cpu()


        for index,i  in enumerate(torch.max(logit, 1)[1].view(targets.size()).data):
            if targets.data[index].item()==0:
                if i.item()==0:
                    tn+=1
                if i.item()==1:
                    fn+=1
            else:
                if i.item() == 0:
                    fp += 1
                if i.item() == 1:
                    tp += 1
        # print()
    size = len(test_id)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation epoch({})- loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(0,
                                                                        avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    print("tn,fn,fp,tp")
    print(str(tn)+" - "+str(fn)+" - "+str(fp)+" - "+str(tp))
    steps=0
    # sess = 'train'

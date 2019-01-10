import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V


class textCNN_Kim(nn.Module):
    #model = textCNN_Kim(input_width=1,input_dim=input_dim=300,class_num=2,kernel_num=200,kernel_sizes=[2,3,4,5,6,7])
    def __init__(self,input_width,input_dim,class_num,kernel_num,kernel_sizes):
        super(textCNN_Kim,self).__init__()
        self.convs1 = nn.ModuleList([nn.Conv2d(input_width, kernel_num, (K, input_dim)) for K in kernel_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(kernel_sizes)*kernel_num, class_num)

    def forward(self, x):
        # x = Variable(x)

        x=V(x)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        print("-----------")
        print(x[0].size())
        print(x[0].size(2))
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)    # 在给定维度上对输入的张量序列进行连接操作，1： - - - -
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit
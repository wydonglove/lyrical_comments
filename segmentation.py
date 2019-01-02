import pandas as pd
from pyhanlp import *

dframe = pd.read_excel('data/pos.xlsx')
#print(dframe.isna())

absolute_pos_contents = dframe.query('归口=="社会维稳"')
pos_contents = dframe.query('归口.str.contains("社会维稳")')
neg_contents = dframe.query('~归口.str.contains("社会维稳")')
print(pos_contents)
absolute_pos_contents.to_csv("absolute_pos.csv")
neg_contents.to_csv("neg.csv")
pos_contents.to_csv("pos.csv")

poss = pos_contents['内容']
negs = neg_contents['内容']

#vec_path1 ='d:/data/wordVec/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'


def clean(str):
    str=str.replace('<br>','')
    str=str.replace(' ','')
    return str

Segment = JClass("com.hankcs.hanlp.seg.Segment")
Term = JClass("com.hankcs.hanlp.seg.common.Term")
segment = HanLP.newSegment().enableOrganizationRecognize(False).enablePlaceRecognize(False).enableNameRecognize(True)
vecs_=[]
vecs2_=[]
for pos in poss:
    vecs_.append(segment.seg(clean(pos)).toString())
for neg in negs:
    vecs2_.append(segment.seg(clean(neg)).toString())

with open('seg_pos.txt','w',encoding='utf-8') as f:
    for vec in vecs_:
        f.write(vec)
        f.write('\n')
f.close()
with open('seg_neg.txt','w',encoding='utf-8') as f:
    for vec in vecs2_:
        f.write(vec)
        f.write('\n')
f.close()
print('completed..')
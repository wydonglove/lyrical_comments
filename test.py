# # -*- coding:utf-8 -*-
# Author：wancong
# Date: 2018-04-30
from pyhanlp import *


def demo_organization_recognition():
    """ 机构名识别
    [我/rr, 在/p, 上海/ns, 林原科技有限公司/nt, 兼职/vn, 工作/vn, ，/w]
    [我/rr, 经常/d, 在/p, 台川喜宴餐厅/nt, 吃饭/vi, ，/w]
    [偶尔/d, 去/vf, 开元地中海影城/nt, 看/v, 电影/n, 。/w]
    """

    sentences = [
      "我在上海林原科技有限公司兼职工作，",
        "我经常在台川喜宴餐厅吃饭，",
        "偶尔去开元地中海影城看电影。",
        ]
    Segment = JClass("com.hankcs.hanlp.seg.Segment")
    Term = JClass("com.hankcs.hanlp.seg.common.Term")

    segment = HanLP.newSegment().enableOrganizationRecognize(True)
    for sentence in sentences:
        term_list = segment.seg(sentence)
        print(term_list)


if __name__ == "__main__":
    #import doctest
    #doctest.testmod(verbose=True)
    #demo_organization_recognition()
    import torch
    from torch.autograd import Variable

    ##单位矩阵来模拟输入
    input = torch.ones(1, 1, 5, 5)
    #input = torch.ones(5, 5)
    print(input)
    input = Variable(input)
    x = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, groups=1)
    out = x(input)
    print(out)
    print("----------")
    print(list(x.parameters()))
    f_p = list(x.parameters())[0]
    print("************:{}".format(f_p))
    f_p = f_p.data.numpy()
    print(f_p[0])
    print("the result of first channel in image:", f_p[0].sum() + (0.2306))
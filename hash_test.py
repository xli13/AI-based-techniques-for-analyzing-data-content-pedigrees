import sys
from minhash import MinHash
import os
import pickle as pkl
import numpy as np
import jieba
import re

def fingerprint(content="",n_grams=3,min_length = 50,hash_num = 256):
    def split_line(line):
        split_line = []
        i = 0
        if len(line) < n_grams:
            split_line.append(line.encode("utf8"))
        else:
            while i <= len(line) - n_grams:
                split_line.append(line[i:i + n_grams].encode("utf8"))  # todo 新增encode("utf8)
                i += 1
        return split_line

    content_split = split_line(content)
    mh = MinHash(num_perm=hash_num)
    mh.update_batch(content_split)
    digest = mh.digest()
    return digest

input_text="""凌云项目,河南省大冶县闵行淮安街I座,50,研发部,555.0,474,C开发,250
九方项目,辽宁省南宁县长寿董街w座,50,研发部,140.0,110,前端开发,250
昊嘉项目,台湾省关岭市涪城太原街a座,50,销售部,1857.0,0.0,销售经理,200
开发区世创项目,湖北省想市城北北京路Y座,50,研发部,1749.0,0.0,Java开发,50
七喜项目,黑龙江省玉英市滨城福州街w座,50,研发部,1196.0,0.0,前端开发,300
浦华众城项目,云南省梅市长寿黄街I座,50,销售部,93.0,0.0,销售经理,250
济南亿次元项目,海南省济南市永川潜江路b座,50,销售部,247.0,208,销售经理,250
九方项目,山西省华县淄川王街P座,50,研发部,1784.0,0.0,Java开发,250
七喜项目,贵州省林县白云合山路h座,50,销售部,1455.0,0.0,销售经理,150
立信电子项目,安徽省秀芳县翔安香港路I座,50,研发部,430.0,361,Java开发,350
精芯项目,浙江省齐齐哈尔市翔安广州路L座,50,研发部,227.0,0.0,前端开发,350
MBP软件项目,江西省斌县南湖永安路c座,50,销售部,216.0,147,销售经理,350
"""
input_text_finger=fingerprint(input_text)

corpus_finger=pkl.load(open("corpus_finger.pkl","rb"))
source_map=pkl.load(open("source_map.pkl","rb"))

sim_list=[]
# 找到最大的相似度
for corpu_fg in corpus_finger:
    comm_num = 0
    for fd, bd in zip(corpu_fg, input_text_finger):
        if fd == bd:
            comm_num += 1
    score = comm_num / len(input_text_finger)
    sim_list.append(score)

sim_list=np.array(sim_list)
preds=np.argsort(-sim_list)[:10]

for pred in preds:
    for fn,i in source_map.items():
        if i==pred:
            print("最相近的结果为：",fn, "得分为：", sim_list[pred])

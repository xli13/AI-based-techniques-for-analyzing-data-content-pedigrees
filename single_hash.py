import sys
from minhash import MinHash
import os
import pickle as pkl
import jieba

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

train_dir="train"
train_files=os.listdir(train_dir)

# 读入训练数据
corpus_finger=[]
source_map={}
for i,fn in enumerate(train_files):
    print(i,fn)
    lines=[]
    with open(os.path.join(train_dir,fn),encoding='utf-8') as f:
        next(f)
        next(f)
        for line in f:
            lines.append(line)
    total_line="".join(lines)
    finger=fingerprint(total_line)
    corpus_finger.append(finger)
    source_map[fn]=i

pkl.dump(corpus_finger,open("corpus_finger.pkl","wb"))
pkl.dump(source_map,open("source_map.pkl","wb"))

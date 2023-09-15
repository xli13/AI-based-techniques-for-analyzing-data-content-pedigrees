import os
import sys
from gensim import corpora, models, similarities
from gensim import corpora
import jieba
import pickle as pkl
from gensim.models.word2vec import Word2Vec

train_dir="train"
train_files=os.listdir(train_dir)
# 读入训练数据
corpus=[]
source_map={}
for i,fn in enumerate(train_files):
    lines=[]
    with open(os.path.join(train_dir,fn)) as f:
        next(f)
        next(f)
        for line in f:
            line=line.strip()
            line = list(jieba.cut(line))
            lines.extend(line)
    corpus.append(lines)
    source_map[fn]=i

# 将文档向量化
dictionary = corpora.Dictionary(corpus)
bow_corpus = [dictionary.doc2bow(text) for text in corpus]

pkl.dump(corpus,open("corpus.pkl","wb"))
pkl.dump(dictionary,open("dictionary.pkl","wb"))
pkl.dump(source_map,open("source_map.pkl","wb"))
pkl.dump(bow_corpus,open("bow_corpus.pkl","wb"))

# 将dtfidf提取特征
tfidf = models.TfidfModel(bow_corpus)
pkl.dump(tfidf,open("tfidf_ model.pkl","wb"))

# 训练词向量
model = Word2Vec(corpus, workers=4, size=16)
model.init_sims(replace=True)
model.save("./word2vec.bin")

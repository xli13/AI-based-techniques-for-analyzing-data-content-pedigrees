import os
import sys
from gensim import corpora, models, similarities
from gensim import corpora
import jieba
import pickle as pkl
from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from crypt import encrypt_encrypt

corpus=pkl.load(open("corpus.pkl","rb"))
bow_corpus=pkl.load(open("bow_corpus.pkl","rb"))
dictionary=pkl.load(open("dictionary.pkl","rb"))
word2vec_model = Word2Vec.load("./word2vec.bin")
tfidf_model = pkl.load(open("./tfidf_ model.pkl","rb"))

corpus_vec=[]
for bow_line in bow_corpus:
    tfidf_line=dict(tfidf_model[bow_line])
    doc_vec=np.zeros(16)
    for wi,c in bow_line:
        word=dictionary[wi]
        if word in word2vec_model.wv and wi in tfidf_line:
            ratio=tfidf_line[wi]
            doc_vec+=word2vec_model.wv[word]*ratio
    corpus_vec.append(doc_vec)
corpus_vec=np.array(corpus_vec)

input_text="""山东省,汉族,"城市探索,王者荣耀",15743127350,19920921,邮政工程,群众
山东省,汉族,"国学,哲学",14725357306,19840908,电子商务,群众
山东省,汉族,咖啡,15575190687,19881004,海洋科学,群众
西藏自治区,壮族,"垂钓,城市探索,休闲运动",13440027262,19880614,设施农业科学与工程,党员
山东省,汉族,"读书会,动画电影,室内装修",15707010220,19921020,汉语言文学,群众
山东省,汉族,缝纫,15762504573,19960301,电信工程及管理,党员
香港特别行政区,汉族,"辩论,战争史",18161724777,19871016,智能建造与智慧交通,群众
"""

input_text = list(jieba.cut(input_text))
bow_line=dictionary.doc2bow(input_text)
tfidf_line=dict(tfidf_model[bow_line])
doc_vec=np.zeros(16)
for wi,c in bow_line:
    word=dictionary[wi]
    if word in word2vec_model.wv and wi in tfidf_line:
        ratio=tfidf_line[wi]
        doc_vec+=word2vec_model.wv[word]*ratio

# 计算相似度排名
score=cosine_similarity(corpus_vec,doc_vec.reshape(1, -1))
# print(score)
pred=score.argmax(0)[0]

source_map=pkl.load(open("source_map.pkl","rb"))
for fn,i in source_map.items():
    if i==pred:
        print("最相近的结果为：",fn)

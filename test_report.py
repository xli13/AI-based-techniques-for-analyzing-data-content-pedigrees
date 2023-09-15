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

#计算所有文档向量
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


# 加密并解密
# corpus_vec=encrypt_encrypt(corpus_vec)

#改变输入
# inputx=[["x","x"]]
# corpus=inputx

##########  开始模拟溯源过程: 舍弃一半
count=0
for i,doc in enumerate(corpus):
    half_doc=doc[:int(len(doc)/2)]
    bow_line=dictionary.doc2bow(half_doc)
    tfidf_line=dict(tfidf_model[bow_line])
    doc_vec=np.zeros(16)
    for wi,c in bow_line:
        word=dictionary[wi]
        if word in word2vec_model.wv and wi in tfidf_line:
            ratio=tfidf_line[wi]
            doc_vec+=word2vec_model.wv[word]*ratio

    # 计算相似度排名
    score=cosine_similarity(corpus_vec,doc_vec.reshape(1, -1))
    pred=score.argmax(0)[0]
    if pred==i:
        count+=1
print("舍弃一半溯源准确率为：",count/len(corpus))


########## 开始模拟溯源过程: 同义词替换
count=0
ratio_replace=0.1
for i,doc in enumerate(corpus):
    wors_list=[]
    most_similar_list=[]
    for w in doc[:2000]:
        if w not in word2vec_model.wv:
            continue
        wors_list.append(w)
    random_idx=np.random.permutation(len(wors_list))[:int(len(wors_list)*ratio_replace)]
    for idx in random_idx:
        w=wors_list[idx]
        most_similar_word=word2vec_model.wv.most_similar(positive=[w], topn=1)[0][0]
        wors_list[idx]=most_similar_word

    bow_line=dictionary.doc2bow(wors_list)
    tfidf_line=dict(tfidf_model[bow_line])
    doc_vec=np.zeros(16)
    for wi,c in bow_line:
        word=dictionary[wi]
        if word in word2vec_model.wv and wi in tfidf_line:
            ratio=tfidf_line[wi]
            doc_vec+=word2vec_model.wv[word]*ratio

    # 计算相似度排名
    score=cosine_similarity(corpus_vec,doc_vec.reshape(1, -1))
    pred=score.argmax(0)[0]
    if pred==i:
        count+=1

print("随机相似替换一部分溯源准确率为：",count/len(corpus))

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import hashlib
from sklearn.metrics import mutual_info_score

d1=pd.read_csv('结果/TF表.csv',encoding='GBK')
del d1['个人-姓名']
d2=d1.copy()
for each in d2.columns[1:]:
    d2[each]=[each2.replace("T",'1').replace("F",'0') for each2 in d2[each]]
    d2[each]=d2[each].astype(int)
rr=[]
for each in d2.columns[1:]:
    a=d2[each].tolist()
    r=[]
    for each2 in d2.columns[1:]:
        if each2 !=each:
            b=d2[each2].tolist()
            mi=mutual_info_score(a, b)
            r.append(mi)
    rr.append([each,np.mean(r)])
mi=pd.DataFrame(rr).sort_values(by=1)[::-1]
mi.columns=['属性','MI']
mi.to_excel('结果/2-Mutual information selects.xlsx',index=False)


#定义UID
r=[]
i=1
for each in tqdm(os.listdir('data')):
    if '.csv' in each:
        df=pd.read_csv('data/'+each,header=1)
        tu=tuple([each for each in df.columns])
        hash_values=hash(tu)
        r.append([each,hash_values])
uid=pd.DataFrame(r)
uid.columns=['文档','UID']
uid.to_excel('结果/UID.xlsx',index=False)
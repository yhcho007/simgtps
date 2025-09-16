import faiss, pickle
import numpy as np
from typing import List, Dict
class FaissDB:
    def __init__(self, dim=384, index_path=None):
        self.dim=dim
        self.index = faiss.IndexFlatIP(dim)
        self.meta=[]
        if index_path:
            self.load(index_path)
    def add(self, vectors, metas):
        v = np.array(vectors).astype('float32')
        faiss.normalize_L2(v)
        self.index.add(v)
        self.meta.extend(metas)
    def search(self, qvec, k=5):
        q = np.array(qvec).astype('float32')
        faiss.normalize_L2(q)
        D,I = self.index.search(q,k)
        res=[]
        for il,dl in zip(I,D):
            hits=[]
            for idx,score in zip(il,dl):
                if 0<=idx<len(self.meta):
                    m=dict(self.meta[idx]); m['score']=float(score); hits.append(m)
            res.append(hits)
        return res
    def save(self,path):
        faiss.write_index(self.index,path+'.index')
        with open(path+'.meta','wb') as f: pickle.dump(self.meta,f)
    def load(self,path):
        self.index = faiss.read_index(path+'.index')
        with open(path+'.meta','rb') as f: self.meta=pickle.load(f)

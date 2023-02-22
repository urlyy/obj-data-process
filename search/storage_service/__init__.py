__all__ = ["Type", "insert", "select"]

import copy
import os
from enum import Enum
import faiss
import pickle
import atexit
import numpy as np

import search

dir_path = os.path.join(os.getcwd(),"search", "index_data")
if os.path.exists(dir_path) is False:
    os.mkdir(dir_path)


class Type(Enum):
    IMG2IMG = "IMG2IMG",
    TXT2IMG = "TXT2IMG",
    TXT2VIDEO = "TXT2VIDEO",
    VIDEO2VIDEO = "VIDEO2VIDEO"

index4TxtSearchImg = None
index4ImgSearchImg = None
index4TxtSearchVideo = None
index4VideoSearchVideo = None
# clip_vit_base_patch32 512,1
# resnet50 2048,1
# clip_vit_b32 512,1
# x3d_m 2048,1

# PCA 降维
# IVF 倒排
# PQ 乘积量化
# IVFPQ 先倒排再量化
class Index4TxtSearchImg:
    dimension = 512

    def __init__(self, bucket_id: int):
        self.index_path = os.path.join(dir_path, f"img_search_by_txt-{bucket_id}.index")
        # self.idmap_path = os.path.join(dir_path, f"img_search_by_txt-{bucket_id}_map.pkl")
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            # with open(self.idmap_path, 'rb') as f:
            #     self.idmap = pickle.load(f)
        else:
            global index4TxtSearchImg
            if index4TxtSearchImg is None:
                index4TxtSearchImg = faiss.index_factory(self.dimension, "PCA128,IVF100,PQ16")
                tmp_data = np.random.rand(256, self.dimension).astype('float32')
                index4TxtSearchImg.train(tmp_data)
            self.index = copy.deepcopy(index4TxtSearchImg.index)
            # self.idmap = faiss.IndexIDMap(self.index)

    def add(self, e:np.ndarray, idx:int):
        ids = np.array([idx]).astype('int64')
        self.index.add_with_ids(e, ids)

    def remove(self,idx):
        ids = np.array([idx]).astype('int64')
        self.index.remove_ids(ids)

    def search(self, e, k):
        D,I = self.index.search(e,k)
        return I[0]

    def __del__(self):
        faiss.write_index(self.index,self.index_path)


class Index4ImgSearchImg:
    dimension = 2048

    def __init__(self, bucket_id: int):
        self.index_path = os.path.join(dir_path, f"img_search_by_img-{bucket_id}.index")
        # self.idmap_path = os.path.join(dir_path, f"img_search_by_img-{bucket_id}_map.pkl")
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            global index4ImgSearchImg
            if index4ImgSearchImg is None:
                index4ImgSearchImg = faiss.index_factory(self.dimension, "PCA128,IVF100,PQ16")
                tmp_data = np.random.rand(256, self.dimension).astype('float32')
                index4ImgSearchImg.train(tmp_data)
            self.index = copy.deepcopy(index4ImgSearchImg.index)

    def add(self, e: np.ndarray, idx: int):
        ids = np.array([idx]).astype('int64')
        self.index.add_with_ids(e, ids)

    def remove(self, idx):
        ids = np.array([idx]).astype('int64')
        self.index.remove_ids(ids)

    def search(self, e, k):
        D, I = self.index.search(e, k)
        return I[0]

    def __del__(self):
        faiss.write_index(self.index, self.index_path)


class Index4TxtSearchVideo:
    dimension = 512

    def __init__(self, bucket_id: int):
        self.index_path = os.path.join(dir_path, f"video_search_by_txt-{bucket_id}.index")
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            global index4TxtSearchVideo
            if index4TxtSearchVideo is None:
                index4TxtSearchVideo = faiss.index_factory(self.dimension, "PCA128,IVF100,PQ16")
                tmp_data = np.random.rand(256, self.dimension).astype('float32')
                index4TxtSearchVideo.train(tmp_data)
            self.index = copy.deepcopy(index4TxtSearchVideo.index)


    def add(self, e: np.ndarray, idx: int):
        ids = np.array([idx]).astype('int64')
        self.index.add_with_ids(e, ids)

    def remove(self, idx):
        ids = np.array([idx]).astype('int64')
        print(type(self.index))
        self.index.remove_ids(ids)

    def search(self, e, k):
        D, I = self.index.search(e, k)
        return I[0]

    def __del__(self):
        faiss.write_index(self.index, self.index_path)


class Index4VideoSearchVideo:
    dimension = 2048

    def __init__(self, bucket_id: int):
        self.index_path = os.path.join(dir_path, f"video_search_by_video-{bucket_id}.index")
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            global index4VideoSearchVideo
            if index4VideoSearchVideo is None:
                index4VideoSearchVideo = faiss.index_factory(self.dimension, "PCA128,IVF100,PQ16")
                tmp_data = np.random.rand(256, self.dimension).astype('float32')
                index4VideoSearchVideo.train(tmp_data)
            self.index = copy.deepcopy(index4VideoSearchVideo.index)

    def add(self, e: np.ndarray, idx: int):
        ids = np.array([idx]).astype('int64')
        self.index.add_with_ids(e, ids)

    def remove(self, idx):
        ids = np.array([idx]).astype('int64')
        self.index.remove_ids(ids)

    def search(self, e, k):
        D, I = self.index.search(e, k)
        return I[0]

    def __del__(self):
        faiss.write_index(self.index, self.index_path)


def get_idx(t:Type,bucket_id: int):
    idx = None
    if t == Type.TXT2IMG:
        idx = Index4TxtSearchImg(bucket_id)
    elif t == Type.IMG2IMG:
        idx = Index4ImgSearchImg(bucket_id)
    elif t == Type.TXT2VIDEO:
        idx = Index4TxtSearchVideo(bucket_id)
    elif t == Type.VIDEO2VIDEO:
        idx = Index4VideoSearchVideo(bucket_id)
    return idx



def insert(embeddings, t: Type, bucket_id: int,object_id:int):
    idx = get_idx(t,bucket_id)
    idx.add(embeddings,object_id)

def remove( t: Type, bucket_id: int,object_id:int):
    idx = get_idx(t,bucket_id)
    idx.remove(object_id)


def select(e:np.ndarray, t: Type, bucket_id: int, k: int) -> list:
    idx = get_idx(t, bucket_id)
    l = idx.search(e, k)
    return l



# 初始时建立建立原型对象
index4TxtSearchImg = get_idx(Type.TXT2IMG,0)
index4ImgSearchImg = get_idx(Type.IMG2IMG,0)
index4TxtSearchVideo = get_idx(Type.TXT2VIDEO,0)
index4VideoSearchVideo = get_idx(Type.VIDEO2VIDEO,0)



def cleanup():
    global index4TxtSearchImg,index4ImgSearchImg,index4TxtSearchVideo,index4VideoSearchVideo
    # 执行清理操作
    index4TxtSearchImg = None
    index4ImgSearchImg = None
    index4TxtSearchVideo = None
    index4VideoSearchVideo = None

atexit.register(cleanup)
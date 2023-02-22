# encoding:utf-8
__all__ = ["extract_embeddings"]

import numpy as np
from towhee import dc

def set_dimension(x: np.ndarray) -> np.ndarray:
    x.shape = (x.shape[0], x.shape[1] if len(x.shape) > 1 else 1)
    x = x.reshape(1, x.shape[0])
    return x

def extract_embeddings(img:np.ndarray)->np.ndarray:
    embeddings = \
        dc['img']([img]) \
            .image_embedding.timm['img', 'vec'](model_name='resnet50')\
            .tensor_normalize['vec', 'vec']() \
            .map(lambda x: set_dimension(x.vec)) \
            .to_list()[0]
    return embeddings
   
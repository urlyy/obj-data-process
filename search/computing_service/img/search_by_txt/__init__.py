# encoding:utf-8
__all__ = ["extract_embeddings_img","extract_embeddings_txt"]

import numpy as np
from towhee import dc
import av




def set_dimension(x: np.ndarray) -> np.ndarray:
    x.shape = (x.shape[0], x.shape[1] if len(x.shape) > 1 else 1)
    x = x.reshape(1, x.shape[0])
    return x

def extract_embeddings_txt(txt:str)->np.ndarray:
    embeddings = \
        dc['txt']([txt]) \
            .image_text_embedding.clip['txt', 'vec'](model_name='clip_vit_base_patch32', modality='text') \
            .tensor_normalize['vec', 'vec']() \
            .map(lambda x: set_dimension(x.vec)) \
            .to_list()[0]
    return embeddings

# def add(img_bytes:bytes):
#     dc['img']([img_bytes]) \
#         .runas_op['img', 'img'](func=lambda bytes: cv2.imdecode(np.array(bytearray(bytes), dtype='uint8'), cv2.IMREAD_UNCHANGED)) \

def extract_embeddings_img(img:np.ndarray)->np.ndarray:
    embeddings = \
        dc['img']([img]) \
            .image_text_embedding.clip['img', 'vec'](model_name='clip_vit_base_patch32', modality='image') \
            .tensor_normalize['vec', 'vec']() \
            .map(lambda x: set_dimension(x.vec)) \
            .to_list()[0]
    return embeddings

# def init_data():
#     towhee.read_csv('reverse_image_search.csv') \
#         .runas_op['id', 'id'](func=lambda x: int(x)) \
#         .image_decode['path', 'img']() \
#         .image_text_embedding.clip['img', 'vec'](model_name='clip_vit_base_patch32', modality='image') \
#         .tensor_normalize['vec', 'vec']() \
#         .map(lambda x: set_dimension(x.vec))\
#         .runas_op(func=lambda x: insert(x,Type.Img,"search_by_txt"))
#
# def add(img_bytes:bytes):
#     dc['img']([img_bytes]) \
#         .runas_op['img', 'img'](func=lambda bytes: cv2.imdecode(np.array(bytearray(bytes), dtype='uint8'), cv2.IMREAD_UNCHANGED)) \
#         .image_text_embedding.clip['img', 'vec'](model_name='clip_vit_base_patch32', modality='image') \
#         .tensor_normalize['vec', 'vec']() \
#         .map(lambda x: set_dimension(x.vec)) \
#         .runas_op(func=lambda x: insert(x, Type.Img,"search_by_txt"))
#
# def search(text, top_k) -> list:
#     res = \
#         towhee.dc['text']([text]) \
#             .image_text_embedding.clip['text', 'vec'](model_name='clip_vit_base_patch32', modality='text') \
#             .tensor_normalize['vec', 'vec']() \
#             .map(lambda x: set_dimension(x.vec)) \
#             .runas_op(func=lambda x: select(x,Type.Img,top_k)) \
#             .to_list()
#     return res[0]
#
#
#
#
# from PIL import Image
# import matplotlib.pyplot as plt
#
#
# def show_img(path: str, ) -> None:
#     img = Image.open(fp=path)
#     plt.axis('off')  # 不显示坐标轴
#     plt.imshow(img)  # 将数据显示为图像，即在二维常规光栅上。
#     plt.show()  # 显示图片
#
#
# if __name__ == '__main__':
#     data_file = "reverse_image_search.csv"
#     df = pd.read_csv(data_file, sep=",", names=["id", "path", "label"], skiprows=1)
#     print(df.shape)
#     print("搜索")
#     indexes = search("light", 3)
#     print(indexes)
#     paths = df["path"].iloc[indexes].values
#     label = df["label"].iloc[indexes].values
#     print(paths)
#     print(label)
#     for path in paths:
#         show_img(path)
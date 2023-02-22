# encoding:utf-8
__all__ = ["extract_embeddings_txt","extract_embeddings_video"]

import numpy as np
from towhee import dc

def set_dimension(x: np.ndarray) -> np.ndarray:
    x.shape = (x.shape[0], x.shape[1] if len(x.shape) > 1 else 1)
    x = x.reshape(1, x.shape[0])
    return x

def extract_embeddings_txt(txt:str)->np.ndarray:
    embeddings = dc['text']([txt]) \
        .video_text_embedding.clip4clip['text', 'vec'](model_name='clip_vit_b32', modality='text') \
        .map(lambda x: set_dimension(x.vec)) \
        .to_list()[0]
    return embeddings

def extract_embeddings_video(file_path)->np.ndarray:
    embeddings = dc['path']([file_path]) \
        .video_decode.ffmpeg['path', 'frames'](sample_type='uniform_temporal_subsample', args={'num_samples': 12})\
        .runas_op['frames', 'frames'](func=lambda x: [y for y in x]) \
        .video_text_embedding.clip4clip['frames', 'vec'](model_name='clip_vit_b32', modality='video')\
        .map(lambda x: set_dimension(x.vec))\
        .to_list()[0]
    # embeddings = \
    #     dc['frames']([frames]) \
    #         .runas_op['frames', 'frames'](func=lambda x: [y for y in x]) \
    #         .video_text_embedding.clip4clip['frames', 'vec'](model_name='clip_vit_b32', modality='video') \
    #         .to_list()[0]
    # print(embeddings.frames)
    return embeddings

# towhee.read_csv(test_sample_csv_path)
#       .runas_op['video_id', 'id'](func=lambda x: int(x[-4:]))
#       .video_decode.ffmpeg['video_path', 'frames'](sample_type='uniform_temporal_subsample', args={'num_samples': 12})
#       .runas_op['frames', 'frames'](func=lambda x: [y for y in x])
#       .video_text_embedding.clip4clip['frames', 'vec'](model_name='clip_vit_b32', modality='video', device=device)
#       .to_milvus['id', 'vec'](collection=collection, batch=30)
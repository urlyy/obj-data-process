__all__ = ["extract_embeddings"]

import numpy as np
from towhee import dc


def set_dimension(x: np.ndarray) -> np.ndarray:
    x.shape = (x.shape[0], x.shape[1] if len(x.shape) > 1 else 1)
    x = x.reshape(1, x.shape[0])
    return x

def extract_embeddings(file_path)->np.ndarray:
    e = dc['path']([file_path]) \
        .video_decode.ffmpeg['path', 'frames'](sample_type='uniform_temporal_subsample', args={'num_samples': 16}) \
        .action_classification['frames', ('labels', 'scores', 'vec')].pytorchvideo(model_name='x3d_m', skip_preprocess=True) \
        .map(lambda x: set_dimension(x.vec))\
        .to_list()[0]
    return e
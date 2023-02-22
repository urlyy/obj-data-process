import io
import math
import os
import tempfile
from functools import reduce, partial

import av
import numpy as np
import cv2
from fastapi import UploadFile
import logging

from towhee.types import VideoFrame

logger = logging.getLogger()


def __img2cv2(img: UploadFile) -> np.ndarray:
    file_bytes: bytes = img.file.read()
    img_cv2 = cv2.imdecode(np.array(bytearray(file_bytes), dtype='uint8'), cv2.IMREAD_UNCHANGED)
    return img_cv2


def img2byte(image):
    byte_stream = io.BytesIO()
    image.save(byte_stream, format='PNG')
    return byte_stream.getvalue()


# 从video_decoder包偷过来的
def __video2frames(video: UploadFile, start_time, sample_type=None, args=None):
    def get_video_duration(video):
        if video.duration is not None:
            return float(video.duration * video.time_base)
        elif video.metadata.get('DURATION') is not None:
            time_str = video.metadata['DURATION']
            return reduce(lambda x, y: float(x) * 60 + float(y), time_str.split(':'))
        else:
            return None

    def decdoe(video, container, start_time):
        if start_time is not None:
            start_offset = int(math.floor(start_time * (1 / video.time_base)))
        else:
            start_offset = 0
        seek_offset = start_offset
        seek_offset = max(seek_offset - 1, 0)
        try:
            container.seek(seek_offset, any_frame=False, backward=True, stream=video)
        except av.AVError as e:
            logger.error(
                'Seek to start_time: %s sec failed, the offset is %s, errors: %s' % (start_time, seek_offset, str(e)))
            raise RuntimeError from e
        for frame in container.decode(video):
            if frame.time < start_time:
                continue
            yield frame

    class SAMPLE_TYPE:
        UNIFORM_TEMPORAL_SUBSAMPLE = 'uniform_temporal_subsample'
        TIME_STEP_SAMPLE = 'time_step_sample'

    def _no_sample(frame_iter):
        yield from frame_iter

    def _time_step_sample(frame_iter, start_time, end_time):
        time_step = args.get('time_step')
        if time_step is None:
            raise RuntimeError('time_step_sample sample lost args time_step')

        time_index = start_time
        for frame in frame_iter:
            if time_index >= end_time:
                break

            if frame.time >= time_index:
                time_index += time_step
                yield frame

    def _uniform_temporal_subsample(frame_iter, total_frames):
        num_samples = args.get('num_samples')
        if num_samples is None:
            raise RuntimeError('uniform_temporal_subsample lost args num_samples')

        indexs = np.linspace(0, total_frames - 1, num_samples).astype('int')
        cur_index = 0
        count = 0
        for frame in frame_iter:
            if cur_index >= len(indexs):
                return

            while cur_index < len(indexs) and indexs[cur_index] <= count:
                cur_index += 1
                yield frame
            count += 1

    def get_sample(stream, duration):
        if sample_type is None:
            return _no_sample
        elif sample_type.lower() == SAMPLE_TYPE.UNIFORM_TEMPORAL_SUBSAMPLE:
            end_time = duration
            start_time = 0
            nums = int(stream.average_rate * (end_time - start_time))
            return partial(_uniform_temporal_subsample, total_frames=nums)
        elif sample_type.lower() == SAMPLE_TYPE.TIME_STEP_SAMPLE:
            start_time = 0
            end_time = duration
            return partial(_time_step_sample, start_time=start_time, end_time=end_time)
        else:
            raise RuntimeError('Unkown sample type: %s' % sample_type)

    with av.open(video.file) as container:
        stream = container.streams.video[0]
        duration = get_video_duration(stream)
        if duration is None:
            duration = float(container.duration) / 1000000
        image_format = 'RGB'
        frame_gen = decdoe(stream, container, start_time)
        sample_function = get_sample(stream, duration)
        for frame in sample_function(frame_gen):
            timestamp = int(frame.time * 1000)
            ndarray = frame.to_ndarray(format='rgb24')
            img = VideoFrame(ndarray, image_format, timestamp, frame.key_frame)
            yield img


# 保存为本地文件
def __save_video2local(dir, video: UploadFile):
    if not os.path.exists(dir):
        os.mkdir(dir)
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[-1], dir=dir)
    tf.write(video.file.read())
    # ff.seek(0)  # 从头读取，和一般文件对象不同，seek方法的执行不能少
    return tf

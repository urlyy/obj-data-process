__all__ = ['app', "reduce", "_reduce"]

import io
import os
import tempfile

import librosa
import numpy as np
import soundfile as sf
import torch
from asteroid.models import BaseModel
from fastapi import APIRouter, UploadFile

# 如果不存在则下载模型
model = BaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")
# 用于放大音量
gain = 10 ** (19 / 20)  # 计算增益，将19dB转换为增益因子
dir = os.path.join(os.getcwd(), "noise_reduce", "tmp_file")
if not os.path.exists(dir):
    os.mkdir(dir)

app = APIRouter()


def reduce(bytesIO: io.BytesIO):
    noisy_audio, sr = librosa.load(bytesIO)
    # 放大音量
    noisy_audio = np.multiply(noisy_audio, gain)
    # 数据类型预处理
    noisy_audio = torch.from_numpy(noisy_audio).to(torch.float32)
    # 使用模型进行增强
    with torch.no_grad():
        enhanced_audio = model(noisy_audio)
    return enhanced_audio.squeeze().numpy(), sr


def _reduce(file: UploadFile):
    bytesIO = io.BytesIO(file.file.read())
    return reduce(bytesIO)
    # 读取音频
    # noisy_audio, sr = sf.read(bytesIO)


@app.post("/audio/noisereduce", summary="音频降噪")
async def noise_reduce(file: UploadFile):
    enhanced_audio, sr = _reduce(file)
    # 将增强后的音频保存到文件
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=dir)
    sf.write(tf.name, enhanced_audio.squeeze(), sr)
    tf.close()
    return {"message": "降噪完毕"}

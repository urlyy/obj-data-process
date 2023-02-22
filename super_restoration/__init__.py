__all__=['app']

import os
import tempfile

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from fastapi import UploadFile, APIRouter

from super_restoration.realesrgan import RealESRGANer



# 加载 Real-ESRGAN 模型
model_dir = os.path.join(os.getcwd(),'model')
model_path = os.path.join(os.getcwd(),"super_restoration",'model','RealESRGAN_x4plus.pth')
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
netscale = 4
file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
# 本地无模型文件就下载
if not os.path.isfile(model_path):
    for url in file_url:
        model_path = load_file_from_url(
            url=url, model_dir=model_dir, progress=True, file_name=None)

# 默认参数
dni_weight = None
tile=0
tile_pad=10
pre_pad=0
fp32=True
gpu_id=0
ext = "auto"
# 图片大小的缩放程度
outscale = 1.0
# 处理器
upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=None
)
dir = os.path.join(os.getcwd(),"super_restoration","tmp_file")
if not os.path.exists(dir):
    os.mkdir(dir)

app = APIRouter()

@app.post("/img/restore",summary="画质修复")
def handle(file:UploadFile):
    file.read()
    imgname, extension = os.path.splitext(os.path.basename(file.filename))
    img_np = np.frombuffer(file.file.read(), dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_mode = 'RGBA'
    else:
        img_mode = None
    try:
       output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    else:
        # 导出为什么后缀名
        if ext == 'auto':
            extension = extension[1:]
        else:
            extension = ext
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'
        # 县创建空文件再写入
        tf = tempfile.NamedTemporaryFile(delete=False, suffix="."+extension, dir=dir)
        cv2.imwrite(tf.name, output)
        tf.close()
    return "修复成功"
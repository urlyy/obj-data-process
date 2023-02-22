import os

from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
from fastapi import APIRouter, UploadFile

from utils.util import __save_video2local

app = APIRouter()

# 读入与输出的文件的文件夹
input_dir = os.path.join(os.getcwd(), "keyframe", "input")
out_dir = os.path.join(os.getcwd(), "keyframe", "out")
# 提取的帧的数量
no_of_frames_to_returned = 4
# 一些可复用的对象
vd = Video()
disk_writer = KeyFrameDiskWriter(location=out_dir)


@app.get("/video/frames", summary="获取关键帧")
async def extract(file: UploadFile):
    tf = __save_video2local(input_dir, file)
    print(os.path.exists(tf.name))
    vd.extract_video_keyframes(
        no_of_frames=no_of_frames_to_returned,
        file_path=tf.name,
        writer=disk_writer
    )

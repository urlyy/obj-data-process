__all__ = ['app']

# 创建路由对象
import os
import time

from fastapi import APIRouter, UploadFile, Query, Form
from starlette.background import BackgroundTasks

from search.computing_service import e4img_add, e4img_search_by_txt, e4img_search_by_img, e4video_add, \
    e4video_search_by_txt, e4video_search_by_video
from search.storage_service import insert, Type, select, remove
from utils.util import __img2cv2, __save_video2local

dir = os.path.join(os.getcwd(), "search", "tmp_video")

app = APIRouter()


@app.post("/img/{bucket_id}/{object_id}", summary="添加图片")
async def add_img(bucket_id: int, object_id: int, file: UploadFile):
    img_cv2 = __img2cv2(file)
    res = e4img_add(img_cv2)
    insert(res["e4img_search_by_txt"], Type.TXT2IMG, bucket_id, object_id)
    insert(res["e4img_search_by_img"], Type.IMG2IMG, bucket_id, object_id)
    return {"code": 200, "message": "索引插入图片成功"}


@app.delete("/img/{bucket_id}/{object_id}", summary="删除图片")
async def remove_img(bucket_id: int, object_id: int):
    remove(Type.TXT2IMG, bucket_id, object_id)
    remove(Type.IMG2IMG, bucket_id, object_id)
    return {"code": 200, "message": "索引删除图片成功"}


@app.get("/img/{bucket_id}/txt", summary="文字搜图")
async def search_img_by_text(bucket_id: int, text=Query(None), top_k=Query(None)):
    e = e4img_search_by_txt(text)
    top_k = int(top_k)
    object_id_list = select(e, Type.TXT2IMG, bucket_id, top_k)
    return {
        "code": 200,
        "message": "文字搜索桶内图片成功",
        "data": object_id_list
    }


@app.get("/img/{bucket_id}/img", summary="以图搜图")
async def search_img_by_img(bucket_id: int, file: UploadFile, top_k=Form(None)):
    img_cv2 = __img2cv2(file)
    e = e4img_search_by_img(img_cv2)
    top_k = int(top_k)
    object_id_list = select(e, Type.IMG2IMG, bucket_id, top_k)
    return {
        "code": 200,
        "message": "文字搜索桶内图片成功",
        "data": object_id_list
    }


def print_after_5():
    time.sleep(5)
    print("打印出来了")


@app.get("/img/duplication", summary="图片查重，还没写")
async def img_duplication(file: UploadFile, bt: BackgroundTasks):
    img_cv2 = __img2cv2(file)
    e = e4img_search_by_img(img_cv2)
    bt.add_task(func=print_after_5)
    return {"message": "不知道是插入的时候查还是查的时候再查"}


@app.post("/video/{bucket_id}/{object_id}", summary="添加视频")
async def add_video(bucket_id: int, object_id: int, file: UploadFile):
    tf = __save_video2local(dir, file)
    res: dict = e4video_add(tf.name)
    tf.close()
    os.remove(tf.name)
    insert(res["e4video_search_by_txt"], Type.TXT2VIDEO, bucket_id, object_id)
    insert(res["e4video_search_by_video"], Type.VIDEO2VIDEO, bucket_id, object_id)
    # frames = __video2frames(file,0,sample_type="uniform_temporal_subsample",args={'num_samples': 12})
    # for i in frames:
    #     print(i)
    # # frame = iio.imread(io.BytesIO(file.file.read()), format_hint=".mp4")
    # print(tf.name)
    # res:dict = e4video_add(tf.name)
    # tf.close()  # 关闭文件即删除了文件
    # print(res)
    return {
        "code": 200,
        "message": "索引插入视频成功"
    }


@app.delete("/video/{bucket_id}/{object_id}", summary="删除视频")
async def remove_video(bucket_id: int, object_id: int):
    remove(Type.TXT2VIDEO, bucket_id, object_id)
    remove(Type.VIDEO2VIDEO, bucket_id, object_id)
    return {"code": 200, "message": "索引删除视频成功"}


@app.get("/video/{bucket_id}/txt", summary="文字搜视频")
async def search_video_by_text(bucket_id: int, text=Query(None), top_k=Query(None)):
    e = e4video_search_by_txt(text)
    top_k = int(top_k)
    object_id_list = select(e, Type.TXT2VIDEO, bucket_id, top_k)
    return {
        "code": 200,
        "message": "文字搜索桶内视频成功",
        "data": object_id_list
    }


@app.get("/video/{bucket_id}/video", summary="以视频搜视频")
async def search_video_by_video(bucket_id: int, file: UploadFile, top_k=Form(None)):
    tf = __save_video2local(dir, file)
    e = e4video_search_by_video(tf.name)
    tf.close()
    os.remove(tf.name)
    top_k = int(top_k)
    object_id_list = select(e, Type.VIDEO2VIDEO, bucket_id, top_k)
    return {
        "code": 200,
        "message": "视频搜索桶内视频成功",
        "data": object_id_list
    }


@app.get("/video/duplication", summary="视频查重，还没写")
async def img_duplication(bt: BackgroundTasks):
    # bt.add_task(send_email,email,message=datetime.now().isoformat())
    bt.add_task(func=print_after_5)
    return {"message": "不知道是插入的时候查还是查的时候再查"}

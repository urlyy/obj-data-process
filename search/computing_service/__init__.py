__all__ = ["e4img_add", "e4img_search_by_txt", "e4img_search_by_img", "e4video_add"]

import numpy as np
from towhee.types.image import Image

import translation
from search.computing_service import img
from search.computing_service import video


def e4img_add(img_cv2: np.ndarray) -> dict:
    i = Image(img_cv2, mode="BGR")
    e4img_search_by_txt = img.search_by_txt.extract_embeddings_img(i)
    e4img_search_by_img = img.search_by_img.extract_embeddings(i)
    res = {
        "e4img_search_by_txt": e4img_search_by_txt,
        "e4img_search_by_img": e4img_search_by_img
    }
    return res


def e4video_add(file_path) -> dict:
    e4video_search_by_txt = video.search_by_txt.extract_embeddings_video(file_path)
    e4video_search_by_video = video.search_by_video.extract_embeddings(file_path)
    res = {
        "e4video_search_by_txt": e4video_search_by_txt,
        "e4video_search_by_video": e4video_search_by_video
    }
    return res


def e4img_search_by_txt(text: str) -> np.ndarray:
    text = translation.translate2en(text)
    e = img.search_by_txt.extract_embeddings_txt(text)
    return e


def e4img_search_by_img(img_cv2: np.ndarray) -> np.ndarray:
    e = img.search_by_img.extract_embeddings(img_cv2)
    return e


def e4img_detect_duplication():
    pass


def e4video_search_by_txt(text: str):
    text = translation.translate2en(text)
    e = video.search_by_txt.extract_embeddings_txt(text)
    return e


def e4video_search_by_video(file_path: str):
    e = video.search_by_video.extract_embeddings(file_path)
    return e

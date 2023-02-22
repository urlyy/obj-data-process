# pip install googletrans==4.0.0-rc1
from fastapi import APIRouter, Form
from googletrans import Translator

app = APIRouter()

translator = Translator()

def translate2en(text)->str:
    s = translator.translate(text, dest='en').text
    return s

@app.get("/txt/translation",summary="文本翻译为英文")
async def translate(text: str = Form("")):
    s = translate2en(text)
    return {
        "code":200,
        "message":"翻译成功",
        "data":s
    }
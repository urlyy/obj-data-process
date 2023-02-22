from fastapi.openapi.docs import get_swagger_ui_html
from fastapi import FastAPI,applications
import os
import uvicorn
import warnings

# from keyframe import app as keyframe_app

from txt2voice import app as txt2voice_app
from noise_reduce import  app as noise_reduce_app


from search import app as search_app
from super_restoration import app as super_restoration_reduce_app
from translation import app as translate_app

app = FastAPI(
    title="OSS-图像处理服务",
    version="1.0.0",
    description="全部接口",
    openapi_url="/api/api.json",
    docs_url="/docs"
)

@app.get("/")
async def hello():
    return {"message": "Hello World", "swagger-url": "ip:port/docs"}



def config_env():
    # 关掉一个警告
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # 修改模型位置
    os.environ['TORCH_HOME'] = os.path.join("..",".model")
    # 修改towhee文件位置
    os.environ['USERPROFILE'] = os.path.join("..")
    # 修改模型路径
    os.environ['TRANSFORMERS_CACHE'] = os.path.join('..',"huggingface",".cache")
    os.environ['ASTEROID_CACHE'] = os.path.join('..', "huggingface", ".cache")
    # 关掉警告
    warnings.simplefilter("ignore")
    # warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    # warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

    # 解决无法访问Swagger的问题
    def swagger_monkey_patch(*args, **kwargs):
        return get_swagger_ui_html(
            *args, **kwargs,
            swagger_js_url='https://cdn.bootcdn.net/ajax/libs/swagger-ui/4.10.3/swagger-ui-bundle.js',
            swagger_css_url='https://cdn.bootcdn.net/ajax/libs/swagger-ui/4.10.3/swagger-ui.css'
        )

    applications.get_swagger_ui_html = swagger_monkey_patch

def mount():
    # app.include_router(keyframe_app,tags=["视频关键帧提取"])
    app.include_router(search_app,tags=["跨模态检索"])
    app.include_router(txt2voice_app, tags=["文本转语音"])
    app.include_router(noise_reduce_app,tags=["音频降噪"])
    app.include_router(super_restoration_reduce_app, tags=["画质修复"])
    app.include_router(translate_app, tags=["文本翻译"])

if __name__ == '__main__':
    config_env()
    mount()
    uvicorn.run(app, port=9010)
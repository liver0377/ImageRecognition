from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import base64
import hashlib
import redis
from io import BytesIO
from PIL import Image
from imageRecognition import predict_general, predict_botany

app = FastAPI()
cache = redis.StrictRedis(host="localhost", port=8888, db=0)


def generate_cache_key(form_data):
    # Generate a unique cache key based on form data
    form_hash = hashlib.sha256(str(form_data).encode()).hexdigest()
    return f"myapp:{form_hash}"


def base64_to_image(base64_str: str) -> Image.Image:
    byte_data = base64.b64decode(base64_str)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    return img


@app.post("/image/general")
async def predict_image(image: str = Form(...), baike_num: int = Form(...)):
    try:
        # 检查缓存
        cache_key = generate_cache_key(image + str(baike_num))
        cached_response = cache.get(cache_key)
        if cached_response:
            return JSONResponse(content=cached_response, status_code=200)

        # 1. 解析请求体
        image_code = image

        # 2. base64解码
        image = base64_to_image(image_code)

        # 3. 进行预测
        predict_results = await predict_general(image, baike_num)

        result = []
        for i in range(baike_num):
            print("result:", result)
            keyword = predict_results[i][0]
            score: float = float(predict_results[i][1])
            image_url = predict_results[i][2]
            baike_info = {
                "image_url": image_url,
                "baike_url": "xxx",
                "description": "xxx",
            }
            result.append(
                {
                    "keyword": keyword,
                    "root": "通用",
                    "score": score,
                    "baike_info": baike_info,
                }
            )

        # 设置缓存
        cache.setex(cache_key, 3600, str({"result": result}))

        return JSONResponse(content={"result": result}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error msg:": str(e)}, status_code=500)


@app.post("/image/botany")
async def predict_image_botany(image: str = Form(...), baike_num: int = Form(...)):
    try:
        # 检查缓存
        # cache_key = generate_cache_key(image + str(baike_num))
        # cached_response = cache.get(cache_key)
        # if cached_response:
        #     return JSONResponse(content=cached_response, status_code=200)

        # 1. 解析请求体
        image_code = image

        # 2. base64解码
        image = base64_to_image(image_code)

        # 3. 进行预测
        predict_results = await predict_botany(image, baike_num)

        result = []
        for i in range(baike_num):
            print("result:", predict_results)
            keyword = str(predict_results[i][0])
            score: float = float(predict_results[i][1])
            image_url = predict_results[i][2]
            baike_info = {
                "image_url": image_url,
                "baike_url": "xxx",
                "description": "xxx",
            }
            result.append(
                {
                    "keyword": keyword,
                    "root": "水果",
                    "score": score,
                    "baike_info": baike_info,
                }
            )
        # 设置缓存
        # cache.setex(cache_key, 3600, str({"result": result}))

        return JSONResponse(content={"result": result}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error msg:": str(e)}, status_code=500)

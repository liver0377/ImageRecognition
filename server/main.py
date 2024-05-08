from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
from imageRecognition import predict_general

app = FastAPI()


class RequestModel(BaseModel):
    image: str
    baike_num: int


def base64_to_image(base64_str: str) -> Image.Image:
    # base64_data = base64_str.split(",")[1]  # Remove the data:image/...;base64, prefix
    byte_data = base64.b64decode(base64_str)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    return img


@app.post("/image/general")
async def predict_image(image: str = Form(...), baike_num: int = Form(...)):
    try:
        # 1. 解析请求体
        image_code = image 

        # 2. base64解码
        image = base64_to_image(image_code)

        # 3. 进行预测
        predict_results = await predict_general(image, baike_num)
        
        result = []
        for i in range(baike_num):
            keyword = predict_results[i][0]
            score : float = float(predict_results[i][1])
            image_url = predict_results[i][2]
            baike_info = {"image_url": image_url, "baike_url": "xxx", "description": "xxx"}
            result.append({"keyword": keyword, "root": "通用", "score": score, "baike_info": baike_info})
            print(f'range: {i} success')

        # results = ["OK"] 
        return JSONResponse(content={"result": result}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error msg:": str(e)}, status_code=500)

@app.post("/image/botany")
async def predict_image_botany(image: str = Form(...), baike_num: int = Form(...)):
    try:
        # 1. 解析请求体
        image_code = image 

        # 2. base64解码
        image = base64_to_image(image_code)

        # 3. 进行预测
        predict_results = await predict_general(image, baike_num)
        
        result = []
        for i in range(baike_num):
            keyword = predict_results[i][0]
            score : float = float(predict_results[i][1])
            image_url = predict_results[i][2]
            baike_info = {"image_url": image_url, "baike_url": "xxx", "description": "xxx"}
            result.append({"keyword": keyword, "root": "通用", "score": score, "baike_info": baike_info})
            print(f'range: {i} success')

        # results = ["OK"] 
        return JSONResponse(content={"result": result}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error msg:": str(e)}, status_code=500)
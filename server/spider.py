import requests
from bs4 import BeautifulSoup
import json
import re


def get_image_url(json_file):
    if "abstractAlbum" not in json_file:
        return None

    if "coverPic" not in json_file["abstractAlbum"]:
        return None

    if "url" not in json_file["abstractAlbum"]["coverPic"]:
        return None

    return json_file["abstractAlbum"]["coverPic"]["url"]


async def get_baidu_baike_image_url(keyword):
    # 构建百度百科搜索 URL
    search_url = f"https://baike.baidu.com/item/{keyword}"

    # 发送 HTTP 请求
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, "html.parser")

    script_elements = soup.find_all("script")

    image_url = None
    for script in script_elements:
        script_string = script.get_text()
        if "PAGE_DATA" in script_string:
            match = re.search(r"\{.*\}", script_string)
            if match:
                json_data = match.group()
                try:
                    parsed_json = json.loads(json_data)
                except json.JSONDecodeError:
                    print("无法解析 JSON 数据。请检查字符串是否有效。")

                image_url = get_image_url(parsed_json)
                print("成功爬取图片url")
            else:
                print("未找到JSON数据")

    return image_url
